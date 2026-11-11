"""Worker-only GPU mesh renderer and orbit camera."""
import ctypes
import numpy as np


class SurfaceGL:
    def __init__(self,engine):
        from OpenGL.GL.shaders import compileShader,compileProgram
        self.engine=engine;self.gl=g=engine.gl;self.yaw=-30.;self.pitch=25.;self.zoom=1.;self.pan=np.zeros(2);self.drag=None;self.count=0;self.side=False
        self.program=compileProgram(compileShader('''#version 330 core
layout(location=0) in vec3 position;layout(location=1) in vec3 normal;
layout(location=2) in vec4 colour;layout(location=3) in float resultOnly;
uniform mat4 transform;uniform mat3 rotation;uniform bool original;
out vec4 tint;
void main(){gl_Position=transform*vec4(position,1);float light=.35+.65*abs(dot(normalize(rotation*normal),normalize(vec3(.3,.5,1))));tint=vec4(colour.rgb*light,colour.a);if((original&&resultOnly>.5)||(!original&&resultOnly<-.5))tint.a=0.;}
''',g.GL_VERTEX_SHADER),compileShader('''#version 330 core
in vec4 tint;out vec4 colour;void main(){if(tint.a<=0.)discard;colour=tint;}
''',g.GL_FRAGMENT_SHADER))
        self.vao=g.glGenVertexArrays(1);self.vbo=g.glGenBuffers(1);g.glBindVertexArray(self.vao);g.glBindBuffer(g.GL_ARRAY_BUFFER,self.vbo)
        for location,size,offset in ((0,3,0),(1,3,12),(2,4,24),(3,1,40)):
            g.glEnableVertexAttribArray(location);g.glVertexAttribPointer(location,size,g.GL_FLOAT,False,44,ctypes.c_void_p(offset))

    def upload(self,vertices,side):
        g=self.gl;self.side=side;self.vertices=vertices.copy();self.count=len(vertices)

    def rotation(self):
        y,p=np.deg2rad([self.yaw,self.pitch]);cy,sy=np.cos(y),np.sin(y);cp,sp=np.cos(p),np.sin(p)
        return np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]],np.float32)

    def draw(self,width,height):
        g=self.gl;r=self.rotation();g.glUseProgram(self.program);g.glBindVertexArray(self.vao);g.glBindBuffer(g.GL_ARRAY_BUFFER,self.vbo)
        # Back-to-front triangles for alpha blending, independently of frame I/O.
        if self.count:
            triangles=self.vertices.reshape(-1,3,11);depth=(triangles[:,:,:3].mean(axis=1)@r.T)[:,2]
            data=np.ascontiguousarray(triangles[np.argsort(depth)].reshape(-1,11));g.glBufferData(g.GL_ARRAY_BUFFER,data.nbytes,data,g.GL_STREAM_DRAW)
        g.glEnable(g.GL_BLEND);g.glBlendFunc(g.GL_SRC_ALPHA,g.GL_ONE_MINUS_SRC_ALPHA);g.glDisable(g.GL_DEPTH_TEST)
        g.glUniformMatrix3fv(g.glGetUniformLocation(self.program,'rotation'),1,g.GL_TRUE,r)
        panels=2 if self.side else 1
        for i in range(panels):
            w=width//panels;g.glViewport(i*w,0,w,height)
            m=np.eye(4,dtype=np.float32);m[:3,:3]=r;m[0,:3]*=min(1,height/w)*.72*self.zoom;m[1,:3]*=min(1,w/height)*.72*self.zoom;m[2,:3]*=-.25;m[:2,3]=self.pan
            g.glUniformMatrix4fv(g.glGetUniformLocation(self.program,'transform'),1,g.GL_TRUE,m)
            g.glUniform1i(g.glGetUniformLocation(self.program,'original'),self.side and i==0);g.glDrawArrays(g.GL_TRIANGLES,0,self.count)
        g.glDisable(g.GL_BLEND);g.glViewport(0,0,width,height)

    def mouse(self,name,x,y,button):
        if name=='button_press_event':self.drag=(x,y,button);return
        if name=='button_release_event':self.drag=None;return
        if self.drag:
            px,py,b=self.drag;dx,dy=x-px,y-py
            if b==1:self.yaw+=dx*.4;self.pitch=np.clip(self.pitch+dy*.4,-89,89)
            else:self.pan+=np.array([dx,-dy])/max(self.engine.window.height(),1)*2
            self.drag=(x,y,b);self.engine.redraw()

    def close(self):
        g=self.gl;g.glDeleteBuffers(1,[self.vbo]);g.glDeleteVertexArrays(1,[self.vao]);g.glDeleteProgram(self.program)
