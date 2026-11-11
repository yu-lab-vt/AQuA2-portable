"""Headless scientific preview export using the same data model as the GUI."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .project import Project
from .viewer_model import layer,curve_series,color,rising_image,plot_curve


def export_preview(project,path,*,mode='Event',channel='channel1',scope='local',frame=0,z=0,selected=(0,),projection=False):
    project=Project.open(project) if isinstance(project,(str,Path)) else project
    image=project.projection(channel,z) if projection else project.frame(frame,z,channel)
    rgba,_=layer(project,mode,channel,scope,frame,z,selected=selected,all_time=projection);rgba[:,:,3]*=.65
    figure=Figure(figsize=(11,7),dpi=150,facecolor='white');FigureCanvasAgg(figure)
    grid=figure.add_gridspec(2,2,height_ratios=[3,1.15],left=.055,right=.98,top=.9,bottom=.09,hspace=.28,wspace=.06)
    for i in range(2):
        ax=figure.add_subplot(grid[0,i]);ax.imshow(image,cmap='gray',origin='lower',interpolation='nearest')
        if i==0:ax.imshow(rgba,origin='lower',interpolation='nearest')
        ax.set_axis_off();ax.set_title(('CFU' if mode=='CFU' else 'Events')+' + overlay' if i==0 else ('Mean projection' if projection else 'Stored raw intensity'),fontsize=10)
    ax=figure.add_subplot(grid[1,:]);kind='CFU' if mode=='CFU' else 'Event'
    infos=project.cfus(channel) if mode=='CFU' else None
    for index,x,y,occurrences in curve_series(project,mode,channel,scope,selected[:8]):
        plot_curve(ax,index,x,y,record=infos[index] if infos is not None else None,label=f'{kind} {index+1}')
    if infos is not None:ax.set_title('Thick: member events | Gray: other events | Thin: inactive',fontsize=8,loc='left')
    if ax.lines:ax.legend(fontsize=8,loc='upper right')
    ax.axvline(frame*float(project.opts['frameRate']),color='#64748b',linewidth=.7,linestyle='--');ax.set_xlabel('Time (s)');ax.set_ylabel('dF/F');ax.grid(alpha=.2)
    summary=project.summary();figure.suptitle(f"AQuA2 Python | {summary['title']} | CH{channel[-1]} | {project.shape[3]} frames",fontsize=12,color='#075a91')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);figure.savefig(path,dpi=150)
    return path
