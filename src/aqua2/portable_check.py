"""Opt-in portable executable verification; writes only to a specified test folder."""
import hashlib,json,os,queue,sys,time,traceback
from pathlib import Path
import numpy as np
from numba import njit


@njit(cache=True)
def _jit_probe(values):return (values*values).sum()


def fingerprint(project):
    result={}
    for c in project.channels:
        result[c]={'events':len(project.events(c)),'cfus':len(project.cfus(c))}
        digest=hashlib.sha256()
        for event in project.events(c):digest.update(np.asarray(event,dtype='<i8').tobytes());digest.update(b'|')
        result[c]['event_indices_sha256']=digest.hexdigest()
        f=project.features(c)
        if f:result[c]['dff_sha256']=hashlib.sha256(np.asarray(f['dffMat'],dtype='<f4').tobytes()).hexdigest()
    return result


def verify(config_path):
    config=json.loads(config_path.read_text(encoding='utf-8'));out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'frozen':bool(getattr(sys,'frozen',False)),'checks':{},'errors':[]};started=time.perf_counter()
    def save():
        report['seconds']=time.perf_counter()-started;(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def fail(error):report['errors'].append(traceback.format_exc() if sys.exc_info()[0] else repr(error));save()
    try:
        from .native import dtw_edge,bilco
        from .se_stats import order_table
        from .workflow_steps import load_input
        from .project import export_project
        from .worker_client import JobClient
        from .viewer import Workbench
        from .viewer_volume import VolumeScene
        assert _jit_probe(np.arange(100,dtype=np.float32))==328350
        assert len(_jit_probe.nopython_signatures)>0,'Numba JIT must be enabled'
        report['checks']['jit_nopython_signatures']=len(_jit_probe.nopython_signatures)
        cuts=dtw_edge(np.zeros((4,4)));assert cuts.size==3
        reference=np.asfortranarray([[0.,1,2,1],[0.,1,2,1]])
        path,_=bilco(reference,reference,np.array([[0,1]]),.1);assert path.shape==(2,3)
        mus,cov=order_table();assert len(mus)>0 and len(cov)>0
        report['checks']['jit_native_resources']=True;save()
        projects=[]
        if config.get('analysis'):
            client=JobClient();client.start({'operation':'run','source':config['tiff'],'config':{'gui_median':True}})
            deadline=time.monotonic()+900;analysis=None;last_error=None
            while time.monotonic()<deadline:
                try:message=client.events.get(timeout=1)
                except queue.Empty:continue
                if message['type']=='complete':analysis=message.get('analysis')
                elif message['type']=='error':last_error=message.get('error')
                elif message['type']=='exit':
                    if message['code']!=0 or analysis is None:raise RuntimeError(last_error or 'Compute worker did not return results')
                    break
            else:client.cancel();raise TimeoutError('Compute verification timed out')
            report['analysis']=fingerprint(analysis);save()
            export=export_project(analysis,out/'export');assert (export/'export_metadata.json').is_file()
            assert not list(export.glob('*.mat'))
            saved=analysis.save(out/'verified.aqua.h5');assert fingerprint(saved)==fingerprint(analysis)
            report['checks']['compute_spawn_export_save']=True;projects.append(('analysis',analysis))
        if config.get('tiff'):projects.append(('raw',load_input(config['tiff'])))
        if config.get('volume'):projects.append(('volume',load_input(config['volume'])))
        if not projects:raise ValueError('Supply a TIFF or a volume for verification')
        class VerifyWindow(Workbench):
            def __init__(self):
                self.expected=None;self.queued=False;self.done=False
                super().__init__();self.geometry('1500x960+20+20');self.after(1500,self.next_project);self.after(120000,lambda:self.finish(TimeoutError('Window verification timed out')))
            def next_project(self):
                if not projects:self.finish();return
                name,p=projects.pop(0);self.expected=(name,p);self.queued=False;self.current_operation='open';self.install_project(p)
            def draw_image(self,request,*arrays):
                super().draw_image(request,*arrays)
                if self.expected and request.project is self.expected[1] and not self.queued:
                    self.queued=True;self.after(150,lambda:self.check(request,arrays))
            def check(self,request,arrays):
                try:
                    name,p=self.expected;gpu=self.gpu_canvas
                    if config.get('require_gpu',True):assert gpu and gpu.process.is_alive(),self.status.get()
                    if gpu:
                        assert gpu.process.pid!=os.getpid();assert not any(k.startswith(('PyQt6.QtCore','PyQt6.QtGui','PyQt6.QtWidgets')) for k in sys.modules)
                        first=gpu.redraw(capture=True)
                        from PIL import Image
                        Image.fromarray(first).save(out/(name+'.png'))
                        if name=='volume':
                            assert isinstance(arrays[0],VolumeScene)
                            gpu._rpc('camera',yaw=55,pitch=40,zoom=1.3);turned=gpu.redraw(capture=True);assert np.any(first!=turned)
                            Image.fromarray(turned).save(out/'volume-rotated.png')
                        report['checks'][name]={'shape':list(p.shape),'gpu':gpu.renderer,'separate_gpu_process':True}
                    else:report['checks'][name]={'shape':list(p.shape),'gpu':False}
                    save();self.expected=None;self.after(150,self.next_project)
                except Exception as error:self.finish(error)
            def finish(self,error=None):
                if self.done:return
                self.done=True
                if error:fail(error)
                else:report['passed']=True;save()
                self.playing=False;self.renderer.close();self.job.cancel();self.io.shutdown(wait=False,cancel_futures=True)
                if self.gpu_canvas:self.gpu_canvas.close()
                self.destroy()
        VerifyWindow().mainloop()
    except Exception as error:fail(error)
    save();return 0 if report['passed'] else 1
