"""Bounded latest-frame preparation and reusable desktop canvas artists."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue,time
from functools import lru_cache
import numpy as np
from numba import njit
from .viewer_model import label_layer,rising_image
from .viewer_display import DisplaySettings,display_gray


@dataclass(frozen=True)
class FrameRequest:
    project:object
    t:int
    z:int
    channel:str
    scope:str
    mode:str
    view:str
    selected:tuple
    visible:tuple
    all_time:bool
    gamma:float
    opacity:float
    overlay:bool
    side:bool
    y_up:bool
    mask_revision:int
    auto_contrast:bool=False
    annotate:bool=False
    label_font:int=10
    display:DisplaySettings|None=None
    volume:bool=False
    surface_level:float=.25
    z_scale:float=1.
    volume_style:str='surface'
    gradient_opacity:float=.3


class FrameReader:
    """Worker-owned read-only HDF handles, with a bounded chunk cache.

    Scientific ArrayRef/Project APIs remain untouched. A render worker reuses
    its handles instead of reopening a movie (and losing its chunk cache) on
    every frame. Changing project closes every handle before new reads.
    """
    def __init__(self):self.project=None;self.files={}
    def close(self):
        for file in self.files.values():file.close()
        self.files.clear();self.project=None
    def _source(self,project,value):
        from .project_io import ArrayRef
        if self.project is not project:self.close();self.project=project
        if not isinstance(value,ArrayRef):return value
        import h5py
        if value.path not in self.files:self.files[value.path]=h5py.File(value.path,'r',rdcc_nbytes=32*1024**2,rdcc_nslots=20011)
        return self.files[value.path][value.node]
    def frame(self,project,t,z,channel,kind='stored'):
        from .project_io import ArrayRef
        movie=project.data['movies'][channel].get(kind)
        if movie is None:
            if kind=='dF':raise ValueError('This project does not contain a dF movie')
            movie=project.movie(channel)
        source=self._source(project,movie)
        return source[t,z,:,:].T if isinstance(movie,ArrayRef) and movie.matlab else np.asarray(source[:,:,z,t])
    def reconstruction(self,project,spatial,t,z):
        from .reconstruction import Reconstruction
        from .project_io import ArrayRef
        if spatial is None:return None
        value=spatial.get('datR')
        if value is None or not value.size:return None
        if isinstance(value,Reconstruction):
            pixels=self._source(project,value.pixels);values=self._source(project,value.values)
            h,w,depth,_=value.shape;start=(t*depth+z)*h*w
            def search(target):
                lo=0;hi=pixels.shape[0]
                while lo<hi:
                    mid=(lo+hi)//2
                    if int(pixels[mid])<target:lo=mid+1
                    else:hi=mid
                return lo
            lo=search(start);hi=search(start+h*w);out=np.zeros((h,w),dtype=value.dtype,order='F')
            out.ravel(order='F')[np.asarray(pixels[lo:hi],np.int64)-start]=values[lo:hi]
            return out
        source=self._source(project,value)
        return source[t,z,:,:].T if isinstance(value,ArrayRef) and value.matlab else np.asarray(source[:,:,z,t])


def intensity_limits(project,channel,kind='stored'):
    options=project.opts;number=channel[-1]
    if kind=='dF':return 0.,float(options.get('maxdF'+number,1.))
    movie=project.movie(channel);dtype=np.dtype(movie.dtype)
    default=(0.,float(np.iinfo(dtype).max)) if dtype.kind in 'ui' else (0.,1.)
    low=float(options.get('minValueDat'+number,default[0]));high=float(options.get('maxValueDat'+number,default[1]))
    display=project.data.get('display',{}).get('matlab_scale',{})
    return low+(high-low)*float(display.get('min',0.)),low+(high-low)*float(display.get('max',1.))


@lru_cache(maxsize=16)
def _gray_lut(dtype,low,high,gamma):
    values=np.arange(np.iinfo(np.dtype(dtype)).max+1,dtype=float)
    scaled=np.clip((values-low)/max(high-low,1e-12),0,1)
    if gamma!=1:scaled=scaled**gamma
    return np.floor(scaled*255+.5).astype(np.uint8)


def grayscale_u8(array,limits,gamma=1.,auto=False):
    low,high=limits;gamma=max(.2,gamma)
    if auto:
        finite=array[np.isfinite(array)]
        low,high=np.percentile(finite,[.5,99.5]) if finite.size else (0,1)
        if high<=low:high=low+1
    if array.dtype==np.uint8 and low==0 and high==255 and gamma==1:return array
    if array.dtype in (np.dtype('uint8'),np.dtype('uint16')):
        gray=_gray_lut(array.dtype.str,float(low),float(high),float(gamma))[array]
    else:
        scaled=np.clip((array.astype(float)-low)/max(high-low,1e-12),0,1)
        if gamma!=1:scaled=scaled**gamma
        gray=np.floor(np.nan_to_num(scaled)*255+.5).astype(np.uint8)
    return gray


def grayscale_rgb(array,limits,gamma=1.,auto=False):
    return np.repeat(grayscale_u8(array,limits,gamma,auto)[...,None],3,axis=-1)


@njit(cache=True,nogil=True)
def video_pixels(gray,raw,labels,alpha,palette,selected,opacity,overlay,side):
    """One bounded pass; no full-size floating RGB intermediates per frame."""
    h,w=gray.shape;main=np.empty((h,w,3),np.uint8);other=np.empty((h if side else 0,w,3),np.uint8)
    for row in range(h):
        for column in range(w):
            label=labels[row,column];highlight=False
            if overlay and label>0 and label<len(selected) and selected[label]:
                for dr in range(-1,2):
                    for dc in range(-1,2):
                        rr=row+dr;cc=column+dc
                        if rr<0 or rr>=h or cc<0 or cc>=w or labels[rr,cc]!=label:highlight=True
            strength=opacity*(1. if highlight else alpha[row,column]) if overlay else 0.
            for component in range(3):
                colour=(1. if component==0 else .92 if component==1 else .15) if highlight else palette[label,component]
                main[row,column,component]=np.uint8(min(255.,max(0.,np.floor(gray[row,column]+colour*strength*255.+.5))))
                if side:other[row,column,component]=raw[row,column]
    return main,other


def normalized(array,gamma=1):
    finite=array[np.isfinite(array)]
    lo,hi=np.percentile(finite,[.5,99.5]) if len(finite) else (0,1)
    if hi<=lo:hi=lo+1
    return np.clip((array.astype(float)-lo)/(hi-lo),0,1)**max(.2,gamma)


def playback_delay(previous,now,fps):
    """Absolute cadence avoids accumulating Tk timer rounding on every frame.

    When preparation exceeds the budget, resume from now without trying to
    catch up by queuing frames or advancing unseen frame numbers.
    """
    deadline=max(now,previous+1/fps)
    return deadline,max(1,round((deadline-now)*1000))


def annotate_labels(image,hits,project,cfu=False,selected=(),font_size=10):
    """Rasterize small ID glyphs at source resolution, before GPU scaling.

    CFU footprints are static: cache their boundary and label locations.
    Scientific arrays and export preparation (annotate=False) are untouched.
    """
    from PIL import Image,ImageDraw
    cache=project._cache.get('display_id_layout') if cfu else None
    if cache is not None and cache[0] is hits:positions,edge=cache[1:]
    else:
        rows,cols=np.nonzero(hits);labels=hits[rows,cols]
        counts=np.bincount(labels);xs=np.bincount(labels,weights=cols);ys=np.bincount(labels,weights=rows)
        positions=[(int(k),float(xs[k]/counts[k]),float(ys[k]/counts[k])) for k in np.flatnonzero(counts) if k>0]
        edge=np.zeros(hits.shape,bool)
        if cfu:
            padded=np.pad(hits,1)
            for dr in range(3):
                for dc in range(3):edge|=(hits>0)&(hits!=padded[dr:dr+hits.shape[0],dc:dc+hits.shape[1]])
            edges={k:np.flatnonzero(edge&(hits==k)) for k,_,_ in positions}
            layer=project._cache.get('display_cfu_layer')
            if layer is not None and hasattr(project,'cfus'):
                from scipy.ndimage import binary_erosion
                channel,z,visible=layer[0];positions=[];edges={}
                for index,record in enumerate(project.cfus(channel)):
                    if visible is not None and index not in visible:continue
                    weights=record['weightMap'];mask=np.asarray(weights[:,:,z] if len(weights.shape)==3 else weights)>.1
                    rows,cols=np.nonzero(mask)
                    if not len(rows):continue
                    nearest=np.argmin((rows-rows.mean())**2+(cols-cols.mean())**2)
                    positions.append((index+1,float(cols[nearest]),float(rows[nearest])))
                    edges[index+1]=np.flatnonzero(mask&~binary_erosion(mask,structure=np.ones((3,3),bool)))
            edge=edges;project._cache['display_id_layout']=(hits,positions,edge)
    if not positions:return image
    if cfu and selected:
        for index in selected:
            pixels=edge.get(index+1)
            if pixels is not None:image[pixels//hits.shape[1],pixels%hits.shape[1]]=[140,217,255]
    picture=Image.fromarray(image);draw=ImageDraw.Draw(picture)
    font=_id_font(font_size)
    for number,x,y in positions:
        draw.text((x,y),str(number),font=font,anchor='mm',fill=(180,230,255) if cfu else (255,255,255),stroke_width=1,stroke_fill=(0,0,0))
    return np.array(picture)


@lru_cache(maxsize=8)
def _id_font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def frame_source(request,reader=None):
    p=request.project;t=request.t;z=request.z;c=request.channel;view=request.view
    raw=reader.frame(p,t,z,c) if reader else p.frame(t,z,c)
    if view=='Mean Projection':image=p.projection(c,z)
    elif view=='Max Projection':image=p.projection(c,z,True)
    elif view=='dF / σ':image=reader.frame(p,t,z,c,'dF') if reader else p.frame(t,z,c,'dF')
    elif view.startswith('Rising Map'):
        if not request.selected:raise ValueError('Select an event to view its rising map.')
        image=rising_image(p,c,request.scope,request.selected[0],z,int(view.split()[1].rstrip('%')))
    else:image=raw
    return image,raw


def prepare_frame(request,reader=None):
    if request.volume:
        if request.volume_style=='gradient':
            from .viewer_volume import prepare_volume
            return prepare_volume(request,reader)
        from .viewer_surface import prepare_surface
        return prepare_surface(request,reader)
    p=request.project;t=request.t;z=request.z;c=request.channel;view=request.view
    image,raw=frame_source(request,reader)
    hits,alpha,palette=label_layer(p,request.mode,c,request.scope,t,z,all_time=request.all_time,visible=request.visible)
    rising=view.startswith('Rising Map')
    limits=intensity_limits(p,c,'dF' if view=='dF / σ' else 'stored')
    if request.display is not None:
        gray=None if rising else display_gray(image,limits,request.display,volume=p.shape[2]>1)
        raw_gray=display_gray(raw,intensity_limits(p,c),request.display,volume=p.shape[2]>1) if request.side else np.empty((0,0),np.uint8)
    else:
        gray=None if rising else grayscale_u8(image,limits,request.gamma,request.auto_contrast)
        if request.annotate and request.side:raw_gray=grayscale_u8(raw,intensity_limits(p,c),1.,False)
        else:raw_gray=gray if image is raw and not rising else grayscale_u8(raw,intensity_limits(p,c),request.gamma,request.auto_contrast) if request.side else np.empty((0,0),np.uint8)
    overlay=request.overlay and view!='Original' and not rising
    if overlay and request.mode=='Event' and not request.all_time and reader is not None:
        strength=reader.reconstruction(p,p.spatial(c,request.scope),t,z)
        if strength is not None:alpha=alpha*(strength.astype(np.float32)/255)
    if rising:return image,np.repeat(raw_gray[...,None],3,-1) if request.side else None,None,hits
    selected=np.zeros(len(palette),np.bool_)
    for index in request.selected:
        if 0<=index+1<len(selected) and not (request.annotate and request.mode=='CFU'):selected[index+1]=True
    opacity=request.display.overlay_brightness if request.display is not None else request.opacity
    main,other=video_pixels(gray,raw_gray,hits,alpha,palette,selected,opacity,overlay,request.side)
    if request.annotate and overlay:main=annotate_labels(main,hits,p,request.mode=='CFU',request.selected,request.label_font)
    return main,other if request.side else None,None,hits


class LatestRenderer:
    """At most one active calculation and one replaceable pending request.

    submit/poll/invalidate run on the UI thread. Workers only post data;
    they never read Tk state or schedule Tk callbacks.
    """
    def __init__(self,prepare=prepare_frame):
        self.reader=FrameReader();self.prepare=(lambda request:prepare_frame(request,self.reader)) if prepare is prepare_frame else prepare;self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='aqua2-frame')
        self.events=queue.Queue();self.generation=0;self.active=False;self.pending=None;self.closed=False
    def submit(self,request):
        self.generation+=1;self.pending=(self.generation,request,time.perf_counter())
        self._launch();return self.generation
    def _launch(self):
        if self.active or self.pending is None or self.closed:return
        token,request,started=self.pending;self.pending=None;self.active=True
        def calculate():
            try:self.events.put((token,request,self.prepare(request),None,started))
            except Exception as error:self.events.put((token,request,None,error,started))
        self.executor.submit(calculate)
    def poll(self):
        try:result=self.events.get_nowait()
        except queue.Empty:return None
        self.active=False;self._launch()
        return result if result[0]==self.generation else None
    def invalidate(self):
        self.generation+=1;self.pending=None
    def release(self):
        self.invalidate();return self.executor.submit(self.reader.close)
    def close(self):
        self.closed=True;self.invalidate();self.executor.submit(self.reader.close);self.executor.shutdown(wait=False)


class BlitCanvas:
    """Cache static axes/titles; redraw images, masks or cursor after updates.

    A real canvas draw (resize/pan/zoom) refreshes the cache before painting
    animated artists, so DPI/layout changes cannot reuse a stale background.
    """
    def __init__(self,canvas):
        self.canvas=canvas;self.artists=[];self.background=None;self.suspended=False;self.cursor_region=None
        canvas.mpl_connect('draw_event',self._draw)
        canvas.mpl_connect('resize_event',lambda event:self.invalidate())
    def invalidate(self):self.background=None;self.cursor_region=None
    def suspend(self):
        # Matplotlib selectors cache the whole figure and redraw only their
        # own axes. Freeze both comparison images into that background.
        self.suspended=True
        for artist in self.artists:artist.set_animated(False)
        self.invalidate();self.canvas.draw()
    def resume(self):
        self.suspended=False
        for artist in self.artists:artist.set_animated(True)
        self.invalidate();self.paint()
    def set_artists(self,artists):
        for artist in self.artists:artist.set_animated(False)
        self.artists=list(artists)
        for artist in self.artists:artist.set_animated(True)
        self.invalidate()
    def _paint(self):
        for artist in self.artists:
            if artist.get_visible():artist.axes.draw_artist(artist)
    def _draw(self,event):
        if self.canvas.is_saving():self.invalidate();return
        if not self.suspended:
            self.background=self.canvas.copy_from_bbox(self.canvas.figure.bbox);self._paint()
            from matplotlib.lines import Line2D
            if len(self.artists)==1 and isinstance(self.artists[0],Line2D):self.cursor_region=self._cursor_box(self.artists[0])
    def _cursor_box(self,cursor):
        from matplotlib.transforms import Bbox
        x=float(cursor.get_transform().transform((cursor.get_xdata()[0],0))[0]);axis=cursor.axes.bbox
        margin=max(3.,cursor.get_linewidth()*self.canvas.figure.dpi/72+2)
        return Bbox.intersection(Bbox.from_extents(np.floor(x-margin),np.floor(axis.y0)-2,np.ceil(x+margin),np.ceil(axis.y1)+2),self.canvas.figure.bbox)
    def paint_cursor(self,cursor):
        if self.background is None or self.suspended:self.paint();return
        current=self._cursor_box(cursor);regions=[box for box in (self.cursor_region,current) if box is not None]
        target=np.asarray(self.canvas.buffer_rgba());source=np.asarray(self.background);height=target.shape[0]
        for box in regions:
            x0,y0,x1,y1=map(int,box.extents);target[height-y1:height-y0,x0:x1]=source[height-y1:height-y0,x0:x1]
        cursor.axes.draw_artist(cursor)
        for box in regions:self.canvas.blit(box)
        self.cursor_region=current
    def paint(self):
        if self.suspended:self.canvas.draw();return
        if self.background is None:self.canvas.draw()
        else:
            self.canvas.restore_region(self.background);self._paint()
            # A maximized figure includes large static gutters. Transfer
            # only the changing axes, not millions of unchanged pixels.
            axes=list(dict.fromkeys(artist.axes for artist in self.artists if artist.get_visible()))
            if axes:
                for axis in axes:self.canvas.blit(axis.bbox)
            else:self.canvas.blit()
