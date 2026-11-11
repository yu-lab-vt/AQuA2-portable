"""Cancelable subprocess jobs; no scientific kernels or blocking reads on Tk."""
from pathlib import Path
import json,os,queue,sys,threading,time


class ProgressState:
    """Current measured step, without inventing a whole-job percentage."""
    def __init__(self,now=None):
        self.started=self.updated=time.monotonic() if now is None else now
        self.message={};self.cpu_seconds=None
    def update(self,message,now=None):
        self.message=message;self.updated=time.monotonic() if now is None else now
    @property
    def fraction(self):
        done=self.message.get('done');total=self.message.get('total')
        if not isinstance(done,(int,float)):return None
        if isinstance(total,(int,float)) and total>0:return max(0,min(1,done/total))
        if total is None and 0<=done<=1:return done
        return None
    @property
    def counts(self):
        done=self.message.get('done');total=self.message.get('total')
        if isinstance(done,(int,float)) and isinstance(total,(int,float)) and total>0:
            return f'{done:,.0f} / {total:,.0f}'
        return ''
    def timing(self,now=None):
        now=time.monotonic() if now is None else now
        def duration(seconds):
            seconds=max(0,int(seconds));return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
        text=f'Elapsed {duration(now-self.started)} · Last update {duration(now-self.updated)}'
        if self.cpu_seconds is not None:text+=f' · CPU time {self.cpu_seconds:.0f} s'
        return text


def _job_worker(job,connection,has_input):
    """Spawned, cancelable compute process; all scientific results stay in memory."""
    import traceback
    from .ipc import send_value,receive_value
    from .workflow import execute_job
    from .project import Project
    lock=threading.Lock();stopped=threading.Event()
    def send(message):
        with lock:connection.send(message)
    def heartbeat():
        while not stopped.wait(5):send({'type':'heartbeat','cpu_seconds':time.process_time()})
    try:
        if has_input:job['project_data']=receive_value(connection,read_only=True)
        threading.Thread(target=heartbeat,daemon=True).start()
        result=execute_job(job,progress=lambda message:send({'type':'progress',**message}),return_project=True)
        stopped.set()
        with lock:
            if isinstance(result,Project):
                connection.send({'type':'result','project_path':str(result.path) if result.path else None})
                start=time.perf_counter();stats=send_value(connection,result.data,shared=os.name=='nt')
                connection.send({'type':'transport','direction':'output','seconds':time.perf_counter()-start,**stats})
            else:connection.send({'type':'complete',**result})
    except BaseException as error:
        stopped.set();send({'type':'error','error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:connection.close()


class _ProcessHandle:
    def __init__(self,process):self.raw=process
    def poll(self):return self.raw.exitcode
    def wait(self):self.raw.join();return self.raw.exitcode
    def terminate(self):self.raw.terminate()


class JobClient:
    def __init__(self):self.events=queue.Queue();self.process=None;self.log_path=None;self.cancelled=False
    @property
    def running(self):return self.process is not None and self.process.poll() is None
    def start(self,job):
        if self.running:raise RuntimeError('A job is already running')
        import multiprocessing,tempfile
        from .ipc import send_value,receive_value
        from .project import Project
        job=dict(job);incoming=job.pop('project_data',None);self.cancelled=False
        descriptor,path=tempfile.mkstemp(prefix='aqua2-job-',suffix='.log');os.close(descriptor);self.log_path=Path(path)
        context=multiprocessing.get_context('spawn');parent,child=context.Pipe(duplex=True)
        if os.name=='nt' and Path(sys.executable).with_name('pythonw.exe').exists():context.set_executable(str(Path(sys.executable).with_name('pythonw.exe')))
        process=context.Process(target=_job_worker,args=(job,child,incoming is not None),daemon=True)
        process.start();child.close();self.process=_ProcessHandle(process);handle=self.process
        def collect():
            try:
                with self.log_path.open('w',encoding='utf-8') as log:
                    if incoming is not None:
                        start=time.perf_counter();stats=send_value(parent,incoming,shared=os.name=='nt')
                        message={'type':'transport','direction':'input','seconds':time.perf_counter()-start,**stats}
                        self.events.put(message);log.write(json.dumps(message)+'\n');log.flush()
                    while True:
                        message=parent.recv()
                        if message.get('type')=='result':
                            data=receive_value(parent)
                            self.events.put({'type':'complete','analysis':Project(data,message.get('project_path'))})
                            log.write('Analysis received in memory; no project auto-save.\n');log.flush()
                        else:
                            log.write(json.dumps(message,ensure_ascii=False)+'\n');log.flush();self.events.put(message)
            except (EOFError,OSError,BrokenPipeError):pass
            except Exception as error:self.events.put({'type':'error','error':str(error)})
            finally:
                parent.close();self.events.put({'type':'exit','code':handle.wait(),'cancelled':self.cancelled,'log':str(self.log_path)})
        threading.Thread(target=collect,daemon=True).start()
    def cancel(self):
        if self.running:self.cancelled=True;self.process.terminate()
