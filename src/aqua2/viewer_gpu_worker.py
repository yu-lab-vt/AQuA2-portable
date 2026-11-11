"""Qt/OpenGL process entry point. Never creates a Tk window or pumps Tcl."""
import ctypes,os,tempfile,traceback
from pathlib import Path
from multiprocessing import shared_memory,parent_process
import numpy as np


class _Renderer:
    def __init__(self,window):
        from OpenGL import GL
        from OpenGL.GL.shaders import compileShader,compileProgram
        from PyQt6.QtGui import QOpenGLContext
        self.window=window;self.gl=GL;self.swaps=0;self.uploads=0;self.texture_shapes={};self.layout=None;self.surface=None;self.surface_active=False;self.renderers={}
        self.context=QOpenGLContext();self.context.setFormat(window.format())
        if not self.context.create() or not self.context.makeCurrent(window):raise RuntimeError('OpenGL 3.3 context creation failed')
        self.renderer=GL.glGetString(GL.GL_RENDERER).decode();self.vendor=GL.glGetString(GL.GL_VENDOR).decode();self.version=GL.glGetString(GL.GL_VERSION).decode()
        if any(word in self.renderer.lower() for word in ('llvmpipe','gdi generic','software rasterizer')):raise RuntimeError('Only a software GL renderer is available: '+self.renderer)
        self.program=compileProgram(compileShader('''#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texcoord;
out vec2 uv;
void main(){gl_Position=vec4(position,0,1);uv=texcoord;}
''',GL.GL_VERTEX_SHADER),compileShader('''#version 330 core
in vec2 uv;
uniform sampler2D image;
out vec4 colour;
void main(){if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1))))discard;colour=texture(image,uv);}
''',GL.GL_FRAGMENT_SHADER))
        self.vao=GL.glGenVertexArrays(1);self.vbo=GL.glGenBuffers(1);GL.glBindVertexArray(self.vao);GL.glBindBuffer(GL.GL_ARRAY_BUFFER,self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER,6*4*4,None,GL.GL_STREAM_DRAW)
        for location,offset in ((0,0),(1,8)):
            GL.glEnableVertexAttribArray(location);GL.glVertexAttribPointer(location,2,GL.GL_FLOAT,False,16,ctypes.c_void_p(offset))
        self.textures=list(map(int,GL.glGenTextures(4)))
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1);GL.glDisable(GL.GL_DITHER);GL.glDisable(GL.GL_FRAMEBUFFER_SRGB);GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self.program);GL.glUniform1i(GL.glGetUniformLocation(self.program,'image'),0)
        self.context.doneCurrent()

    def upload(self,index,array):
        gl=self.gl;h,w,channels=array.shape;gl.glBindTexture(gl.GL_TEXTURE_2D,self.textures[index])
        fmt=gl.GL_RGB if channels==3 else gl.GL_RGBA
        if self.texture_shapes.get(index)!=array.shape:
            gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_NEAREST);gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER,gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_WRAP_S,gl.GL_CLAMP_TO_EDGE);gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_WRAP_T,gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGB8 if channels==3 else gl.GL_RGBA8,w,h,0,fmt,gl.GL_UNSIGNED_BYTE,array);self.texture_shapes[index]=array.shape
        else:gl.glTexSubImage2D(gl.GL_TEXTURE_2D,0,0,0,w,h,fmt,gl.GL_UNSIGNED_BYTE,array)
        self.uploads+=1

    def present(self,message):
        self.surface_active=False
        ratio=self.window.devicePixelRatio()
        self.window.setGeometry(0,0,max(1,round(message['width']/ratio)),max(1,round(message['height']/ratio)))
        if not self.context.makeCurrent(self.window):raise RuntimeError('OpenGL surface lost its context')
        try:
            for descriptor in message['textures']:
                shm=shared_memory.SharedMemory(name=descriptor['name'])
                try:
                    array=np.ndarray(descriptor['shape'],np.uint8,buffer=shm.buf)
                    self.upload(descriptor['index'],array);del array
                finally:shm.close()
            if message['layout'] is not None:self.layout=message['layout']
        finally:self.context.doneCurrent()
        self.window.show();self.redraw()

    def redraw(self,capture=False):
        if self.layout is None and not self.surface_active:return
        if not self.context.makeCurrent(self.window):raise RuntimeError('OpenGL context is unavailable')
        gl=self.gl;ratio=self.window.devicePixelRatio()
        width,height=round(self.window.width()*ratio),round(self.window.height()*ratio)
        try:
            gl.glViewport(0,0,width,height);gl.glDisable(gl.GL_BLEND)
            gl.glClearColor(.929,.945,.961,1);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            if self.surface_active:
                gl.glClearColor(.035,.045,.06,1);gl.glClear(gl.GL_COLOR_BUFFER_BIT);self.surface.draw(width,height)
                pixels=np.frombuffer(gl.glReadPixels(0,0,width,height,gl.GL_RGBA,gl.GL_UNSIGNED_BYTE),np.uint8).reshape(height,width,4)[::-1].copy() if capture else None
                self.context.swapBuffers(self.window);self.swaps+=1;return pixels
            gl.glUseProgram(self.program);gl.glBindVertexArray(self.vao);gl.glBindBuffer(gl.GL_ARRAY_BUFFER,self.vbo);gl.glActiveTexture(gl.GL_TEXTURE0)
            def quad(index,vertices):
                gl.glBindTexture(gl.GL_TEXTURE_2D,self.textures[index]);gl.glBufferSubData(gl.GL_ARRAY_BUFFER,0,vertices.nbytes,vertices);gl.glDrawArrays(gl.GL_TRIANGLES,0,6)
            quad(0,self.layout['full_quad'])
            for i,vertices in enumerate(self.layout['quads']):quad(i+1,vertices)
            gl.glEnable(gl.GL_BLEND);gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA,gl.GL_ONE,gl.GL_ONE_MINUS_SRC_ALPHA);quad(3,self.layout['full_quad']);gl.glDisable(gl.GL_BLEND)
            pixels=None
            if capture:pixels=np.frombuffer(gl.glReadPixels(0,0,width,height,gl.GL_RGBA,gl.GL_UNSIGNED_BYTE),np.uint8).reshape(height,width,4)[::-1].copy()
            self.context.swapBuffers(self.window);self.swaps+=1
            return pixels
        finally:self.context.doneCurrent()

    def close(self):
        self.window.hide()
        if self.context.makeCurrent(self.window):
            for renderer in self.renderers.values():renderer.close()
            gl=self.gl;gl.glDeleteTextures(self.textures);gl.glDeleteBuffers(1,[self.vbo]);gl.glDeleteVertexArrays(1,[self.vao]);gl.glDeleteProgram(self.program);self.context.doneCurrent()
        self.window.destroy()


def gpu_worker(connection,host_id):
    import faulthandler
    log_path=Path(tempfile.gettempdir())/f'aqua2-gpu-worker-{os.getpid()}.log'
    with log_path.open('a',encoding='utf-8') as log:
        faulthandler.enable(file=log,all_threads=True)
        app=None;engine=None;window=None
        try:
            from PyQt6.QtGui import QGuiApplication,QWindow,QSurfaceFormat
            from PyQt6.QtCore import Qt,QTimer
            app=QGuiApplication(['aqua2-gpu-worker']);app.setQuitOnLastWindowClosed(False)
            # Handle all Python callback errors here; never let a virtual Qt
            # callback abort the display process through an uncaught exception.
            def failure(error):
                log.write(str(error)+'\n'+traceback.format_exc());log.flush()
                try:connection.send({'type':'error','error':str(error)})
                except (OSError,EOFError):pass
                app.quit()
            def emit(**event):
                try:connection.send({'type':'input',**event})
                except (OSError,EOFError):app.quit()
            class Surface(QWindow):
                def exposeEvent(self,event):
                    if self.isExposed() and engine is not None:
                        try:engine.redraw()
                        except Exception as error:failure(error)
                def mouse(self,name,event):
                    pos=event.position();ratio=self.devicePixelRatio()
                    button={Qt.MouseButton.LeftButton:1,Qt.MouseButton.MiddleButton:2,Qt.MouseButton.RightButton:3}.get(event.button())
                    if engine is not None and engine.surface_active:
                        try:engine.surface.mouse(name,pos.x(),pos.y(),button)
                        except Exception as error:failure(error)
                        return
                    emit(kind='mouse',name=name,x=pos.x()*ratio,y=pos.y()*ratio,button=button)
                def mousePressEvent(self,event):self.mouse('button_press_event',event)
                def mouseReleaseEvent(self,event):self.mouse('button_release_event',event)
                def mouseMoveEvent(self,event):self.mouse('motion_notify_event',event)
                def wheelEvent(self,event):
                    if engine is not None and engine.surface_active:
                        try:engine.surface.zoom=float(np.clip(engine.surface.zoom*1.12**(event.angleDelta().y()/120),.1,10));engine.redraw()
                        except Exception as error:failure(error)
                def mouseDoubleClickEvent(self,event):
                    if engine is not None and engine.surface_active:
                        try:engine.surface.yaw=-30.;engine.surface.pitch=25.;engine.surface.zoom=1.;engine.surface.pan[:]=0;engine.redraw()
                        except Exception as error:failure(error)
                def keyPressEvent(self,event):
                    key={Qt.Key.Key_Left:'left',Qt.Key.Key_Right:'right',Qt.Key.Key_Space:'space',Qt.Key.Key_Escape:'escape'}.get(event.key())
                    if key:emit(kind='key',key=key)
                    else:event.ignore()
            foreign=QWindow.fromWinId(host_id)
            if foreign is None:raise RuntimeError('Cannot embed GPU window in the Tk canvas')
            fmt=QSurfaceFormat();fmt.setVersion(3,3);fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setSwapInterval(0);fmt.setSamples(0);fmt.setDepthBufferSize(0);fmt.setStencilBufferSize(0)
            window=Surface();window.setFlags(Qt.WindowType.SubWindow|Qt.WindowType.FramelessWindowHint)
            window.setSurfaceType(QWindow.SurfaceType.OpenGLSurface);window.setFormat(fmt);window.setParent(foreign);window.create()
            engine=_Renderer(window)
            def receive():
                try:
                    parent=parent_process()
                    if parent is not None and not parent.is_alive():app.quit();return
                    if not connection.poll():return
                    message=connection.recv();operation=message['operation']
                    if operation=='close':app.quit();return
                    pixels=None
                    if operation=='present':engine.present(message)
                    elif operation=='surface':
                        from .viewer_surface_gl import SurfaceGL
                        ratio=window.devicePixelRatio();window.setGeometry(0,0,max(1,round(message['width']/ratio)),max(1,round(message['height']/ratio)))
                        engine.context.makeCurrent(window)
                        try:
                            old=engine.surface
                            if 'surface' not in engine.renderers:engine.renderers['surface']=SurfaceGL(engine)
                            engine.surface=engine.renderers['surface']
                            if old is not None and old is not engine.surface:
                                for key in ('yaw','pitch','zoom'):setattr(engine.surface,key,getattr(old,key))
                                engine.surface.pan=old.pan.copy()
                            descriptor=message['mesh'];shm=shared_memory.SharedMemory(name=descriptor['name'])
                            try:engine.surface.upload(np.ndarray(descriptor['shape'],np.float32,buffer=shm.buf),message['side'])
                            finally:shm.close()
                            engine.surface_active=True
                        finally:engine.context.doneCurrent()
                        window.show();engine.redraw()
                    elif operation=='volume':
                        from .viewer_volume_gl import VolumeGL
                        from .viewer_volume import VolumeScene
                        ratio=window.devicePixelRatio();window.setGeometry(0,0,max(1,round(message['width']/ratio)),max(1,round(message['height']/ratio)))
                        engine.context.makeCurrent(window);buffers=[]
                        try:
                            old=engine.surface
                            if 'volume' not in engine.renderers:engine.renderers['volume']=VolumeGL(engine)
                            engine.surface=engine.renderers['volume']
                            if old is not None and old is not engine.surface:
                                for key in ('yaw','pitch','zoom'):setattr(engine.surface,key,getattr(old,key))
                                engine.surface.pan=old.pan.copy()
                            arrays=[]
                            for descriptor in message['volumes']:
                                shm=shared_memory.SharedMemory(name=descriptor['name']);buffers.append(shm)
                                arrays.append(np.ndarray(descriptor['shape'],np.float32,buffer=shm.buf))
                            scene=VolumeScene(tuple(arrays[:2]),arrays[2],message['side'],message['z_scale'],message['y_up'],message['opacity'],message['gradient'])
                            engine.surface.upload_volume(scene);engine.surface_active=True
                            # GL has uploaded the arrays; retain only metadata, not shared-memory views.
                            engine.surface.scene=VolumeScene(tuple(np.empty(a.shape,np.uint8) for a in arrays[:2]),None,scene.side,scene.z_scale,scene.y_up,scene.opacity,scene.gradient)
                            del arrays,scene
                        finally:
                            for shm in buffers:shm.close()
                            engine.context.doneCurrent()
                        window.show();engine.redraw()
                    elif operation=='camera':
                        if engine.surface is None:raise ValueError('No surface scene')
                        for key in ('yaw','pitch','zoom'):
                            if key in message:setattr(engine.surface,key,float(message[key]))
                        engine.redraw()
                    elif operation=='redraw':pixels=engine.redraw(message.get('capture',False))
                    elif operation=='hide':window.hide()
                    else:raise ValueError('Unknown GPU display command')
                    camera={key:getattr(engine.surface,key) for key in ('yaw','pitch','zoom')} if engine.surface is not None else None
                    connection.send(dict(type='ack',seq=message['seq'],swaps=engine.swaps,uploads=engine.uploads,pixels=pixels,camera=camera))
                except EOFError:app.quit()
                except Exception as error:failure(error)
            timer=QTimer();timer.timeout.connect(receive);timer.start(1)
            connection.send(dict(type='ready',renderer=engine.renderer,vendor=engine.vendor,version=engine.version,log_path=str(log_path)))
            app.exec()
        except Exception as error:
            log.write(traceback.format_exc());log.flush()
            try:connection.send({'type':'error','error':str(error)})
            except (OSError,EOFError):pass
        finally:
            if engine is not None:
                try:engine.close()
                except Exception:log.write(traceback.format_exc())
            if window is not None:window.setParent(None)
            connection.close();faulthandler.disable()
