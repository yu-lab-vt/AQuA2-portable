"""Tk-side adapter for an isolated Qt/OpenGL display process.

Qt is never imported or pumped in the Tk process. Only display textures
(RGB/RGBA or normalized scalar volumes) reach the worker; scientific movies stay here.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import sys,time
import numpy as np


def image_quad(box,limits,shape,figure_size):
    left,bottom,right,top=box;fw,fh=figure_size;h,w=shape[:2]
    (x0,x1),(y0,y1)=limits
    l,r=2*left/fw-1,2*right/fw-1;b,t=2*bottom/fh-1,2*top/fh-1
    u0,u1=(x0+.5)/w,(x1+.5)/w;v0,v1=(y0+.5)/h,(y1+.5)/h
    return np.array([[l,b,u0,v0],[r,b,u1,v0],[r,t,u1,v1],[l,b,u0,v0],[r,t,u1,v1],[l,t,u0,v1]],np.float32)


class _WindowProxy:
    def __init__(self,adapter):self.adapter=adapter;self.visible=False
    def hide(self):
        if self.visible and not self.adapter.closed:
            self.adapter._rpc('hide');self.visible=False
    def isVisible(self):return self.visible and not self.adapter.closed and self.adapter.process.is_alive()


class GpuCanvas:
    """Synchronous swap ACK before the GUI commits the displayed frame."""
    def __init__(self,owner,canvas):
        self.closed=False;self.after_id=None;self.process=None;self.connection=None
        self.buffers={};self.inputs=[];self.sequence=0;self.window=None
        try:self.initialize(owner,canvas)
        except Exception:
            self.close();raise

    def initialize(self,owner,canvas):
        if sys.platform!='win32':raise RuntimeError('Embedded GPU display requires Windows')
        from .viewer_gpu_worker import gpu_worker
        self.owner=owner;self.canvas=canvas;self.stamp=None;self.blocked=False
        self.error=None;self.swaps=0;self.uploads=0;self.window=_WindowProxy(self)
        context=mp.get_context('spawn');parent,child=context.Pipe()
        self.connection=parent
        self.process=context.Process(target=gpu_worker,args=(child,int(canvas.get_tk_widget().winfo_id())),daemon=True)
        self.process.start();child.close()
        ready=self._receive(20)
        if ready.get('type')!='ready':raise RuntimeError(ready.get('error','GPU worker did not initialize'))
        self.renderer=ready['renderer'];self.vendor=ready['vendor'];self.version=ready['version']
        self.log_path=ready['log_path'];self.pump()

    def _receive(self,timeout):
        import _tkinter
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            # Cross-process SetParent/resize can synchronously notify the Tk
            # host. Service only Tk window events while waiting; never Qt or
            # Tk timer callbacks (which could re-enter frame presentation).
            self.owner.tk.dooneevent(_tkinter.WINDOW_EVENTS|_tkinter.DONT_WAIT)
            if self.connection.poll(min(.005,max(0,deadline-time.monotonic()))):
                try:message=self.connection.recv()
                except EOFError:raise RuntimeError('GPU display process exited unexpectedly') from None
                if message.get('type')=='input':self.inputs.append(message);continue
                if message.get('type')=='error':raise RuntimeError(message['error'])
                return message
            if self.process is not None and not self.process.is_alive():
                raise RuntimeError(f'GPU display process exited (code {self.process.exitcode})')
        raise TimeoutError('GPU display process did not respond in time')

    def _rpc(self,operation,**payload):
        if self.closed:raise RuntimeError('GPU display is closed')
        self.sequence+=1;seq=self.sequence
        self.connection.send(dict(operation=operation,seq=seq,**payload))
        result=self._receive(5)
        if result.get('seq')!=seq:raise RuntimeError('Unexpected GPU display acknowledgement')
        self.swaps=result.get('swaps',self.swaps);self.uploads=result.get('uploads',self.uploads)
        return result

    def _pixels(self,index,array):
        array=np.asarray(array)
        if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2] not in (3,4):raise ValueError('GPU input must be display RGB/RGBA bytes')
        old=self.buffers.get(index)
        if old is None or old[1].shape!=array.shape:
            if old is not None:old[0].close();old[0].unlink()
            shm=shared_memory.SharedMemory(create=True,size=array.nbytes)
            target=np.ndarray(array.shape,np.uint8,buffer=shm.buf);self.buffers[index]=(shm,target)
        shm,target=self.buffers[index];np.copyto(target,array)
        return dict(index=index,name=shm.name,shape=array.shape)

    def supported(self,request):return not self.closed and not self.error and not self.blocked and not request.view.startswith('Rising Map') and not self.owner.draw_kind and not self.owner.navtoolbar.mode

    def layout_stamp(self):
        return (self.owner.display_key,tuple(round(v) for v in self.canvas.figure.bbox.size),
                tuple((tuple(a.get_xlim()),tuple(a.get_ylim()),tuple(a.bbox.extents)) for a in self.owner.image_axes))

    def present(self,request,image,raw):
        width,height=(round(v) for v in self.canvas.figure.bbox.size);stamp=self.layout_stamp()
        textures=[];layout=None
        if stamp!=self.stamp or self.owner.image_blit.background is None:
            self.owner.image_blit.paint();textures.append(self._pixels(0,np.asarray(self.owner.image_blit.background)))
            saved=self.canvas.copy_from_bbox(self.canvas.figure.bbox);self.canvas.get_renderer().clear()
            try:
                images=set(self.owner.image_artists+[self.owner.overlay_artist])
                for artist in self.owner.image_blit.artists:
                    if artist not in images and artist.get_visible():artist.axes.draw_artist(artist)
                overlay=np.asarray(self.canvas.buffer_rgba()).copy()
            finally:self.canvas.restore_region(saved)
            textures.append(self._pixels(3,overlay))
            layout=dict(quads=[image_quad(a.bbox.extents,(a.get_xlim(),a.get_ylim()),image.shape,(width,height)) for a in self.owner.image_axes],
                full_quad=image_quad((0,0,width,height),((-.5,width-.5),(height-.5,-.5)),(height,width),(width,height)))
        textures.append(self._pixels(1,image))
        if request.side:textures.append(self._pixels(2,raw))
        self._rpc('present',width=width,height=height,textures=textures,layout=layout)
        self.stamp=stamp;self.window.visible=True

    def redraw(self,*,capture=False):return self._rpc('redraw',capture=capture).get('pixels')

    def present_surface(self,scene):
        array=scene.vertices;index=4;old=self.buffers.get(index);shape=array.shape
        if old is None or old[1].shape!=shape:
            if old is not None:old[0].close();old[0].unlink()
            shm=shared_memory.SharedMemory(create=True,size=max(1,array.nbytes));self.buffers[index]=(shm,np.ndarray(shape,np.float32,buffer=shm.buf))
        shm,target=self.buffers[index];np.copyto(target,array)
        width,height=(round(v) for v in self.canvas.figure.bbox.size)
        self._rpc('surface',width=width,height=height,mesh=dict(name=shm.name,shape=shape),side=scene.side)
        self.stamp=None;self.window.visible=True

    def present_volume(self,scene):
        arrays=(*scene.volumes,scene.overlay);descriptors=[]
        for index,array in enumerate(arrays,5):
            old=self.buffers.get(index)
            if old is None or old[1].shape!=array.shape:
                if old is not None:old[0].close();old[0].unlink()
                shm=shared_memory.SharedMemory(create=True,size=array.nbytes);self.buffers[index]=(shm,np.ndarray(array.shape,np.float32,buffer=shm.buf))
            shm,target=self.buffers[index];np.copyto(target,array);descriptors.append(dict(name=shm.name,shape=array.shape))
        width,height=(round(v) for v in self.canvas.figure.bbox.size)
        self._rpc('volume',width=width,height=height,volumes=descriptors,side=scene.side,z_scale=scene.z_scale,y_up=scene.y_up,opacity=scene.opacity,gradient=scene.gradient)
        self.stamp=None;self.window.visible=True

    def _dispatch_inputs(self):
        from matplotlib.backend_bases import MouseEvent
        messages,self.inputs=self.inputs,[]
        for message in messages:
            if message['kind']=='mouse':
                event=MouseEvent(message['name'],self.canvas,message['x'],self.canvas.figure.bbox.height-message['y'],button=message['button'])
                self.canvas.callbacks.process(message['name'],event)
            else:
                action={'left':lambda:self.owner.step(-1),'right':lambda:self.owner.step(1),
                    'space':self.owner.toggle_play,'escape':self.owner.cancel_draw}.get(message['key'])
                if action:action()

    def pump(self):
        if self.closed:return
        try:
            if not self.process.is_alive():raise RuntimeError(f'GPU display process exited (code {self.process.exitcode})')
            while self.connection.poll():
                message=self.connection.recv()
                if message.get('type')=='input':self.inputs.append(message)
                elif message.get('type')=='error':raise RuntimeError(message['error'])
                else:raise RuntimeError('Unexpected GPU display message')
            self._dispatch_inputs()
            blocked=bool(self.owner.draw_kind or self.owner.navtoolbar.mode or self.owner.view.get().startswith('Rising Map'))
            if blocked and not self.blocked:self.window.hide();self.owner.image_blit.paint()
            elif self.blocked and not blocked:self.stamp=None;self.owner.request_render()
            if self.stamp is not None and self.layout_stamp()!=self.stamp and not blocked:self.stamp=None;self.owner.request_render()
            self.blocked=blocked;self.after_id=self.owner.after(1 if self.owner.playing else 10,self.pump)
        except Exception as error:self.fail(error)

    def fail(self,error):
        self.error=str(error);self.close();self.owner.use_gpu.set(False)
        self.owner.status.set('GPU worker failed; using fallback: '+str(error))
        self.owner.image_blit.invalidate();self.owner.image_blit.paint();self.owner.request_render()

    def close(self):
        if self.closed:return
        self.closed=True
        if self.after_id:
            try:self.owner.after_cancel(self.after_id)
            except Exception:pass
        if self.connection is not None:
            try:
                if self.process is not None and self.process.is_alive():self.connection.send({'operation':'close','seq':-1})
            except (OSError,EOFError):pass
        if self.process is not None:
            deadline=time.monotonic()+1
            while self.process.is_alive() and time.monotonic()<deadline:
                try:
                    import _tkinter
                    self.owner.tk.dooneevent(_tkinter.WINDOW_EVENTS|_tkinter.DONT_WAIT)
                except Exception:pass
                self.process.join(.01)
            if self.process.is_alive():self.process.terminate();self.process.join(1)
        if self.connection is not None:self.connection.close()
        for shm,array in self.buffers.values():
            shm.close();shm.unlink()
        self.buffers.clear()
