"""Worker-only trilinear GPU ray caster with view-ray gradient opacity."""
import numpy as np
from .viewer_surface_gl import SurfaceGL


class VolumeGL(SurfaceGL):
    def __init__(self,engine):
        from OpenGL.GL.shaders import compileShader,compileProgram
        self.engine=engine;self.gl=g=engine.gl;self.yaw=-30.;self.pitch=25.;self.zoom=1.;self.pan=np.zeros(2);self.drag=None
        self.program=compileProgram(compileShader('''#version 330 core
out vec2 screen;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);screen=p*2.-1.;gl_Position=vec4(screen,0,1);}
''',g.GL_VERTEX_SHADER),compileShader('''#version 330 core
in vec2 screen;out vec4 colour;
uniform sampler3D image;uniform sampler3D labels;
uniform mat3 inverseRotation;uniform vec3 extent;uniform vec3 dimensions;
uniform vec2 scale;uniform vec2 pan;uniform float opacity;uniform float gradient;
uniform float cameraDistance;
uniform bool original;uniform bool yUp;
vec3 texcoord(vec3 p){vec3 t=(p/extent+1.)*.5;t.x=1.-t.x;if(!yUp)t.y=1.-t.y;return t;}
void main(){
    vec3 bg=mix(vec3(.8),vec3(.45),clamp(screen.y*.5+.5,0.,1.));
    vec3 origin=inverseRotation*vec3(0,0,cameraDistance);
    vec3 direction=normalize(inverseRotation*vec3((screen-pan)/scale,-1.));
    vec3 safeDir=sign(direction)*max(abs(direction),vec3(1e-7));safeDir+=vec3(equal(safeDir,vec3(0)))*1e-7;
    vec3 t0=(-extent-origin)/safeDir,t1=(extent-origin)/safeDir;
    vec3 near3=min(t0,t1),far3=max(t0,t1);
    float start=max(max(near3.x,near3.y),near3.z),stop=min(min(far3.x,far3.y),far3.z);
    if(stop<=max(start,0.)){colour=vec4(bg,1);return;}
    start=max(start,0.);
    // Half-voxel sampling with opacity corrected to one voxel of travel.
    vec3 voxelRate=direction/(2.*extent)*dimensions;
    float rate=max(max(abs(voxelRate.x),abs(voxelRate.y)),abs(voxelRate.z));
    float dt=.5/max(rate,1e-6);int count=min(2048,int(ceil((stop-start)/dt)));dt=(stop-start)/float(max(count,1));
    float travel=dt*rate;
    vec4 acc=vec4(0);float previous=0.;
    for(int i=0;i<2048;i++){
        if(i>=count||acc.a>.995)break;
        vec3 pos=origin+direction*(start+(float(i)+.5)*dt);vec3 uv=texcoord(pos);
        float v=texture(image,uv).r;
        // Similar consecutive samples get extra transparency; g=0 is ordinary DVR.
        float edge=clamp(abs(v-previous)/max(travel,1e-5)/max(gradient*.1,1e-6),0.,1.);
        float retained=pow(1.-gradient,8.);
        float a=v*mix(retained,1.,edge)*opacity;
        a=1.-pow(max(1.-a,0.),2.*travel);
        vec4 label=original?vec4(0):texture(labels,uv);
        float la=1.-pow(max(1.-label.a,0.),travel);
        vec3 rgb=vec3(v)*a;
        rgb=label.rgb*la+rgb*(1.-la);a=la+a*(1.-la);
        acc.rgb+=(1.-acc.a)*rgb;acc.a+=(1.-acc.a)*a;previous=v;
    }
    colour=vec4(acc.rgb+(1.-acc.a)*bg,1);
}
''',g.GL_FRAGMENT_SHADER))
        self.vao=g.glGenVertexArrays(1);self.textures=list(map(int,g.glGenTextures(3)));self.shapes={};self.scene=None

    def texture(self,index,array):
        g=self.gl;g.glActiveTexture(g.GL_TEXTURE0+index);g.glBindTexture(g.GL_TEXTURE_3D,self.textures[index])
        data=np.ascontiguousarray(array.transpose(2,0,1,3) if array.ndim==4 else array.transpose(2,0,1),dtype=np.float32)
        d,h,w=data.shape[:3];fmt=g.GL_RGBA if array.ndim==4 else g.GL_RED;internal=g.GL_RGBA16F if array.ndim==4 else g.GL_R32F
        if self.shapes.get(index)!=data.shape:
            for key in (g.GL_TEXTURE_MIN_FILTER,g.GL_TEXTURE_MAG_FILTER):g.glTexParameteri(g.GL_TEXTURE_3D,key,g.GL_LINEAR if index<2 else g.GL_NEAREST)
            for key in (g.GL_TEXTURE_WRAP_S,g.GL_TEXTURE_WRAP_T,g.GL_TEXTURE_WRAP_R):g.glTexParameteri(g.GL_TEXTURE_3D,key,g.GL_CLAMP_TO_EDGE)
            g.glTexImage3D(g.GL_TEXTURE_3D,0,internal,w,h,d,0,fmt,g.GL_FLOAT,data);self.shapes[index]=data.shape
        else:g.glTexSubImage3D(g.GL_TEXTURE_3D,0,0,0,0,w,h,d,fmt,g.GL_FLOAT,data)

    def upload_volume(self,scene):
        self.scene=scene;self.side=scene.side
        for index,array in enumerate((*scene.volumes,scene.overlay)):self.texture(index,array)

    def draw(self,width,height):
        if self.scene is None:return
        g=self.gl;s=self.scene;r=self.rotation();g.glUseProgram(self.program);g.glBindVertexArray(self.vao);g.glDisable(g.GL_BLEND);g.glDisable(g.GL_DEPTH_TEST)
        h,w,d=s.volumes[0].shape;shape=np.array([w,h,d*s.z_scale],np.float32);extent=shape/shape.max()
        uniform=lambda name:g.glGetUniformLocation(self.program,name)
        g.glUniformMatrix3fv(uniform('inverseRotation'),1,g.GL_TRUE,r.T)
        g.glUniform3fv(uniform('extent'),1,extent);g.glUniform3f(uniform('dimensions'),w,h,d)
        g.glUniform1f(uniform('opacity'),s.opacity);g.glUniform1f(uniform('gradient'),s.gradient);g.glUniform1i(uniform('yUp'),s.y_up)
        g.glUniform1f(uniform('cameraDistance'),3./self.zoom)
        g.glUniform2fv(uniform('pan'),1,self.pan);g.glUniform1i(uniform('labels'),2)
        for index in range(3):g.glActiveTexture(g.GL_TEXTURE0+index);g.glBindTexture(g.GL_TEXTURE_3D,self.textures[index])
        panels=2 if s.side else 1
        for i in range(panels):
            pw=width//panels;g.glViewport(i*pw,0,pw,height)
            focal=1./np.tan(np.deg2rad(22.5))
            g.glUniform2f(uniform('scale'),min(1,height/pw)*focal,min(1,pw/height)*focal)
            original=s.side and i==0;g.glUniform1i(uniform('original'),original);g.glUniform1i(uniform('image'),0 if original else 1)
            g.glDrawArrays(g.GL_TRIANGLES,0,3)
        g.glViewport(0,0,width,height)

    def close(self):
        g=self.gl;g.glDeleteTextures(self.textures);g.glDeleteVertexArrays(1,[self.vao]);g.glDeleteProgram(self.program)
