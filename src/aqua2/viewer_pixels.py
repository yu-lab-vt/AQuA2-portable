"""Fast nearest-neighbour display of pre-coloured RGB frames on Agg/TkAgg.

Keep Matplotlib axes and hit coordinates; avoid its repeated float/RGBA
normalisation and resampling of already prepared 8-bit video pixels.
"""
import numpy as np
from PIL import Image
from matplotlib.image import AxesImage
from matplotlib.transforms import IdentityTransform
from matplotlib.backends.backend_agg import RendererAgg


class PixelImage(AxesImage):
    def make_image(self,renderer,magnification=1.,unsampled=False):
        array=self.get_array()
        if (not isinstance(renderer,RendererAgg) or self.figure.canvas.is_saving()
                or unsampled or array is None or array.dtype!=np.uint8 or array.ndim!=3):
            return super().make_image(renderer,magnification,unsampled=unsampled)
        # The axes bbox is the visible viewport. Resample that viewport only,
        # including when zoomed, rather than allocating a giant zoomed image.
        box=self.axes.bbox;left=int(round(box.x0));bottom=int(round(box.y0))
        width=max(1,int(round(box.x1))-left);height=max(1,int(round(box.y1))-bottom)
        inverse=self.axes.transData.inverted()
        x0,y0=inverse.transform((left,bottom));x1,y1=inverse.transform((left+width,bottom+height))
        # Array pixel centres use the same coordinates as imshow. Pillow's
        # extent is top-down; Agg.draw_image receives bottom-up scanlines.
        extent=tuple(round(value,12) for value in (x0+.5,y1+.5,x1+.5,y0+.5))
        image=Image.fromarray(np.asarray(array)).convert('RGBA').transform((width,height),Image.Transform.EXTENT,extent,resample=Image.Resampling.NEAREST,fillcolor=(0,0,0,0))
        rgba=np.asarray(image)[::-1].copy()
        return rgba,left,bottom,IdentityTransform()


def pixel_imshow(ax,array,**kwargs):
    """Equivalent image artist for scalar maps and RGB video frames."""
    ax.set_aspect(kwargs.pop('aspect','equal'));low=kwargs.pop('vmin',None);high=kwargs.pop('vmax',None)
    artist=PixelImage(ax,**kwargs);artist.set_data(array);artist.set_clim(low,high);ax.add_image(artist)
    artist.set_extent(artist.get_extent());ax.autoscale_view();return artist
