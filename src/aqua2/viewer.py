"""Initial local Tk/Matplotlib workbench, following AQuA2's three-column UI."""
import math,queue,time,sys,tkinter as tk
from collections import deque
from tkinter import ttk,filedialog,messagebox
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.widgets import PolygonSelector
from .project import Project
from .viewer_model import event_rows,curve_series,color,polygon_pixels,plot_curve
from .viewer_render import FrameRequest,LatestRenderer,BlitCanvas,playback_delay
from .viewer_pixels import pixel_imshow
from .viewer_display import DisplaySettings,fiji_auto_range
from .worker_client import JobClient,ProgressState
from .viewer_dialogs import NewAnalysisDialog,CFUSettingsDialog,ParameterDialog
from .cfu import default_options
from .pipeline import scalar_options
from .tk_runtime import synchronize_canvas
from .result_export import favorites,EXPORT_CHOICES
from .viewer_step_forms import StepForms,DEFAULT_PARAMETERS
from .workflow_steps import phase,can_detect,display_for,preprocessing_input

matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus']=False


def _number(value):
    try:return f'{float(value):.4g}' if np.isfinite(float(value)) else '—'
    except (ValueError,TypeError):return '—'


def _integer(text):
    value=float(text)
    if not math.isfinite(value) or value!=int(value):raise ValueError('Frame and CFU counts must be integers.')
    return int(value)


def _job_error(message):
    if 'lacks reconstructed datR' in message:
        return 'This MAT lacks datR. Existing features remain available. Rerun event detection for region and landmark features.'
    return message


class Workbench(tk.Tk):
    def __init__(self,path=None,*,image_backend='auto'):
        from .tk_runtime import initialize
        initialize(self);self.title('AQuA2 · Python Workbench');self.configure(background='#e8edf2')
        self.geometry(f'{min(1500,self.winfo_screenwidth()-60)}x{min(960,self.winfo_screenheight()-90)}+20+20');self.minsize(1080,680)
        self.project=None;self.job=JobClient();self.busy=False;self.loading=False;self.dirty=False;self.playing=False;self.selected=[];self.rows=[];self.visible=[];self.hits=None
        self.io=ThreadPoolExecutor(max_workers=1,thread_name_prefix='aqua2-view');self.ui_events=queue.Queue();self.renderer=LatestRenderer();self.polygon=None;self.draw_kind=None;self.completed_path=None;self.last_error=None
        self.play_after=None;self.play_deadline=None;self.ready_render=None;self.present_times=deque(maxlen=60);self.rate_updated=0.;self.requested_frame=0;self.displayed_frame=0;self.mask_revision=0;self.display_key=None;self._updating_slider=False
        self.chain_steps=deque();self.chain_ready=False;self.current_operation=None
        self.gpu_canvas=None;self.use_gpu=tk.BooleanVar(value=image_backend=='qt-gpu' or (image_backend=='auto' and sys.platform=='win32'))
        self.parameter_path=DEFAULT_PARAMETERS;self.parameter_preset=1
        style=ttk.Style(self);style.theme_use('clam');style.configure('.',font=('Microsoft YaHei UI',9))
        style.configure('TFrame',background='#edf1f5');style.configure('TLabelframe',background='#edf1f5');style.configure('TLabel',background='#edf1f5',foreground='#253b50')
        style.configure('TButton',padding=(8,5));style.configure('Primary.TButton',background='#075a91',foreground='white');style.map('Primary.TButton',background=[('active','#0876b9')])
        style.configure('Section.TLabel',background='#075a91',foreground='white',font=('Microsoft YaHei UI',9,'bold'),padding=(8,5))
        style.configure('Task.Horizontal.TProgressbar',background='#1683c4',troughcolor='#d6e2ec',bordercolor='#bdd0e0',lightcolor='#1683c4',darkcolor='#1683c4')
        style.configure('Treeview',rowheight=25,background='white',fieldbackground='white');style.configure('Treeview.Heading',font=('Microsoft YaHei UI',9,'bold'))
        self.channel=tk.StringVar(value='channel1');self.scope=tk.StringVar(value='local');self.mode=tk.StringVar(value='Event');self.view=tk.StringVar(value='Overlay')
        self.frame=tk.IntVar(value=1);self.z=tk.IntVar(value=1);self.fps=tk.IntVar(value=10);self.side=tk.BooleanVar(value=False);self.show_overlay=tk.BooleanVar(value=True)
        self.all_time=tk.BooleanVar();self.y_up=tk.BooleanVar(value=False);self.opacity=tk.DoubleVar(value=.5);self.gamma=tk.DoubleVar(value=1);self.auto_contrast=tk.BooleanVar(value=False);self.actual_fps=tk.StringVar()
        self.display_vars={key:tk.DoubleVar(value=value) for key,value in DisplaySettings().as_dict().items()}
        self.volume_mode=tk.StringVar(value='Z Slice');self.surface_level=tk.DoubleVar(value=.25);self.z_scale=tk.DoubleVar(value=1.);self.gradient_opacity=tk.DoubleVar(value=.3)
        self.auto_divisor=0;self.auto_key=None;self.auto_pending=False;self.display_revision=0
        self.curve_kind=tk.StringVar(value='dF/F');self.cleaned=tk.BooleanVar(value=False);self.favorites_only=tk.BooleanVar();self.search=tk.StringVar()
        self.area_min=tk.StringVar();self.duration_min=tk.StringVar();self.dff_min=tk.StringVar();self.group_filter=tk.StringVar(value='All')
        self.export_choices={key:tk.BooleanVar(value=True) for key in EXPORT_CHOICES}
        self.status=tk.StringVar(value='Load an image or open saved results.');self.summary_text=tk.StringVar(value='No project loaded')
        self.progress_state=ProgressState();self.task_text=tk.StringVar(value='Ready');self.task_timing=tk.StringVar()
        self._build();self.protocol('WM_DELETE_WINDOW',self.close)
        self.bind('<Control-o>',lambda _:self.open_dialog());self.bind('<Control-s>',lambda _:self.save())
        self.bind('<Left>',lambda _:self.step(-1));self.bind('<Right>',lambda _:self.step(1));self.bind('<space>',self.space_key)
        self.after(100,self.poll)
        self.after(8,self.poll_render)
        if self.use_gpu.get():self.after(100,self.toggle_gpu)
        if path:self.after(50,lambda:self.open_project(path))
    def section(self,parent,title):
        panel=ttk.Frame(parent);panel.pack(fill='x',pady=(0,8));ttk.Label(panel,text=title,style='Section.TLabel').pack(fill='x')
        body=ttk.Frame(panel,padding=8);body.pack(fill='x');return body
    def _build(self):
        header=ttk.Frame(self,padding=(10,8));header.pack(fill='x')
        ttk.Label(header,text='AQuA2',font=('Segoe UI',20,'bold'),foreground='#075a91').pack(side='left',padx=(0,10))
        ttk.Label(header,text='Python Workbench',foreground='#64748b').pack(side='left',padx=(0,18))
        for text,command in [('Load Image',self.new_analysis),('Open Project',self.open_dialog),('Import MAT',self.import_dialog),('Save',self.save),('Save As',self.save_as)]:
            ttk.Button(header,text=text,command=command).pack(side='left',padx=3)
        ttk.Button(header,text='Help',command=self.help).pack(side='right')
        task=ttk.Frame(self,padding=(12,6));task.pack(fill='x')
        self.task_label=ttk.Label(task,textvariable=self.task_text,foreground='#075a91',font=('Microsoft YaHei UI',10,'bold'))
        self.task_label.pack(fill='x')
        self.progress=ttk.Progressbar(task,style='Task.Horizontal.TProgressbar',mode='determinate',maximum=100);self.progress.pack(fill='x',pady=(5,3))
        ttk.Label(task,textvariable=self.task_timing,foreground='#61758b').pack(anchor='w')
        task.bind('<Configure>',lambda event:self.task_label.configure(wraplength=max(200,event.width-30)))
        # Reserve the footer before the expandable panes, including at high DPI.
        footer=ttk.Frame(self,padding=(10,5));footer.pack(side='bottom',fill='x')
        ttk.Label(footer,textvariable=self.status,width=1).pack(side='left',fill='x',expand=True)
        self.cancel_button=ttk.Button(footer,text='Cancel Task',command=self.cancel,state='disabled');self.cancel_button.pack(side='right')
        outer=ttk.Panedwindow(self,orient='horizontal');outer.pack(fill='both',expand=True,padx=8,pady=(0,5))
        left=ttk.Frame(outer,width=325);middle=ttk.Frame(outer);right=ttk.Frame(outer,width=330)
        outer.add(left,weight=0);outer.add(middle,weight=1);outer.add(right,weight=0)
        scroll=tk.Canvas(left,width=330,highlightthickness=0,background='#edf1f5');bar=ttk.Scrollbar(left,orient='vertical',command=scroll.yview)
        bar.pack(side='right',fill='y');scroll.pack(side='left',fill='both',expand=True);scroll.configure(yscrollcommand=bar.set)
        content=ttk.Frame(scroll);item=scroll.create_window((0,0),window=content,anchor='nw')
        content.bind('<Configure>',lambda _:scroll.configure(scrollregion=scroll.bbox('all')));scroll.bind('<Configure>',lambda event:scroll.itemconfigure(item,width=event.width))
        self.bind_all('<MouseWheel>',lambda event:scroll.yview_scroll(-int(event.delta/120)*2,'units') if str(event.widget).startswith(str(content)) else None)
        self._left(content);self._center(middle);self._right(right)
    def begin_activity(self,text):
        self.progress_state=ProgressState();self.task_text.set(text);self.task_timing.set('')
        self.progress.stop();self.progress.configure(mode='indeterminate',value=0);self.progress.start(12)
    def toggle_gpu(self):
        self.pause_play()
        if self.gpu_canvas is not None:
            self.gpu_canvas.close();self.gpu_canvas=None
        if self.use_gpu.get():
            try:
                from .viewer_gpu import GpuCanvas
                self.gpu_canvas=GpuCanvas(self,self.image_canvas);self.status.set('GPU enabled: '+self.gpu_canvas.renderer)
            except Exception as error:
                self.use_gpu.set(False);self.status.set('GPU unavailable; using fallback: '+str(error))
        self.image_blit.invalidate();self.request_render()
    def finish_activity(self,text):
        self.progress.stop();self.progress.configure(mode='determinate',value=0);self.task_text.set(text)
        self.task_timing.set(self.progress_state.timing())
    def update_task_progress(self,message):
        self.progress_state.update(message)
        names={'input':'Loading input','preprocessing':'Preprocessing','activity':'Active Regions','detection':'Segmentation / Features','cfu':'CFU Analysis','dependencies':'Dependencies / Groups','features':'Region Features','saving':'Saving project','export':'Export Results','transfer':'Showing results','complete':'Complete'}
        phases={'seeds':'Detecting seeds','seed_prepare':'Preparing multiscale seeds','watershed':'Temporal Segmentation','majority':'Extracting major curves','refine':'Refining events','grow':'Growing events','curve_filter':'Filtering curves','merge_info':'Computing adjacency and delays','merge':'Merging temporal events','spatial':'Spatial Segmentation','evaluate':'Evaluating merges','recompute':'Recomputing curves','Running':'Running','Resuming completed stage':'Resuming completed stage','Writing complete project':'Writing project','Checking source and parameters':'Checking input and parameters'}
        detail=message.get('detail','');fraction=self.progress_state.fraction
        text=names.get(message.get('stage'),message.get('stage',''))+' · '+str(phases.get(detail,detail))
        if fraction is not None:
            if str(self.progress['mode'])!='determinate':self.progress.stop();self.progress.configure(mode='determinate')
            self.progress['value']=100*fraction
            text+=' · Step '+(self.progress_state.counts+' ' if self.progress_state.counts else '')+f'({fraction:.0%})'
        elif str(self.progress['mode'])!='indeterminate':
            self.progress.configure(mode='indeterminate',value=0);self.progress.start(12)
        self.task_text.set(text);self.status.set(text)
    def _left(self,parent):
        body=self.section(parent,'Project / Workflow')
        ttk.Label(body,textvariable=self.summary_text,wraplength=235,justify='left').pack(anchor='w',pady=(0,8))
        self.stage_list=ttk.Label(body,text='Preprocessing → Event Detection → CFU',wraplength=235,foreground='#61758b');self.stage_list.pack(anchor='w',pady=4)
        ttk.Button(body,text='Load Parameters',command=self.load_parameters).pack(fill='x',pady=3)
        self.run_all_button=ttk.Button(body,text='Run All',command=self.run_all,style='Primary.TButton',state='disabled');self.run_all_button.pack(fill='x',pady=3)
        self.step_forms=StepForms(body,self.run_step,self.show_step)
        body=self.section(parent,'Regions / Landmarks')
        row=ttk.Frame(body);row.pack(fill='x');ttk.Button(row,text='Draw Region',command=lambda:self.draw_region('regions')).pack(side='left',expand=True,fill='x',padx=(0,3))
        ttk.Button(row,text='Draw Landmark',command=lambda:self.draw_region('landmarks')).pack(side='left',expand=True,fill='x')
        row=ttk.Frame(body);row.pack(fill='x',pady=3)
        ttk.Button(row,text='Import Region',width=0,padding=(3,5),command=lambda:self.import_masks('regions')).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Import Landmark',width=0,padding=(3,5),command=lambda:self.import_masks('landmarks')).pack(side='left',fill='x',expand=True,padx=(4,0))
        self.region_text=ttk.Label(body,text='Region 0 · Landmark 0');self.region_text.pack(anchor='w',pady=3)
        row=ttk.Frame(body);row.pack(fill='x',pady=3)
        ttk.Button(row,text='Undo Region',command=lambda:self.undo_mask('regions')).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Undo Landmark',command=lambda:self.undo_mask('landmarks')).pack(side='left',fill='x',expand=True,padx=(4,0))
        body=self.section(parent,'Event Filters')
        for label,var in [('Size ≥',self.area_min),('Duration ≥ (s)',self.duration_min),('dF/F ≥',self.dff_min)]:
            row=ttk.Frame(body);row.pack(fill='x',pady=2);ttk.Label(row,text=label,width=13).pack(side='left');ttk.Entry(row,textvariable=var,width=10).pack(side='right',fill='x',expand=True)
        ttk.Button(body,text='Apply Filters',command=self.refresh_table).pack(fill='x',pady=4)
        ttk.Checkbutton(body,text='Favorites Only',variable=self.favorites_only,command=self.refresh_table).pack(anchor='w')
        body=self.section(parent,'Export Results')
        for key,label in EXPORT_CHOICES.items():ttk.Checkbutton(body,text=label,variable=self.export_choices[key]).pack(anchor='w',pady=2)
        ttk.Button(body,text='Export Selected…',command=self.export_dialog,style='Primary.TButton').pack(fill='x',pady=5)
        row=ttk.Frame(body);row.pack(fill='x');ttk.Button(row,text='Task Log',command=self.show_log).pack(side='left',expand=True,fill='x');ttk.Button(row,text='Export View…',command=self.export_figure).pack(side='left',expand=True,fill='x',padx=(4,0))
    def _center(self,parent):
        controls=ttk.Frame(parent,padding=(7,5));controls.pack(fill='x')
        for var,values,width in [(self.mode,['Event','CFU'],6),(self.channel,['channel1'],10),(self.scope,['local','global'],8),(self.view,['Overlay','Original','Mean Projection','Max Projection','dF / σ','Rising Map 20%','Rising Map 50%','Rising Map 80%'],17)]:
            combo=ttk.Combobox(controls,textvariable=var,values=values,width=width,state='readonly');combo.pack(side='left',padx=2);combo.bind('<<ComboboxSelected>>',self.changed)
            if var is self.channel:self.channel_combo=combo
            if var is self.scope:self.scope_combo=combo
            if var is self.view:self.view_combo=combo
        self.channel_combo.pack_forget()
        ttk.Checkbutton(controls,text='Y↑',variable=self.y_up,command=self.request_render).pack(side='left')
        split=ttk.Panedwindow(parent,orient='vertical');split.pack(fill='both',expand=True)
        images=ttk.Frame(split);curves=ttk.Frame(split);split.add(images,weight=4);split.add(curves,weight=1)
        self.image_figure=Figure(figsize=(7,5),dpi=100,facecolor='#edf1f5');self.image_canvas=FigureCanvasTkAgg(self.image_figure,master=images)
        self.image_blit=BlitCanvas(self.image_canvas)
        synchronize_canvas(self.image_canvas)
        self.image_canvas.get_tk_widget().pack(fill='both',expand=True);self.image_canvas.mpl_connect('button_press_event',self.pick_pixel)
        self.image_canvas.mpl_connect('resize_event',lambda _:self.request_render())
        toolbar=NavigationToolbar2Tk(self.image_canvas,images,pack_toolbar=False);toolbar.update();toolbar.pack(fill='x');self.navtoolbar=toolbar
        playback=ttk.Frame(curves,padding=(6,4));playback.pack(fill='x')
        self.play_button=ttk.Button(playback,text='▶ Play',command=self.toggle_play,width=8);self.play_button.pack(side='left')
        ttk.Button(playback,text='‹',width=3,command=lambda:self.step(-1)).pack(side='left',padx=3)
        self.slider=ttk.Scale(playback,from_=1,to=1,command=self.slide);self.slider.pack(side='left',fill='x',expand=True,padx=3)
        ttk.Button(playback,text='›',width=3,command=lambda:self.step(1)).pack(side='left')
        self.frame_entry=ttk.Spinbox(playback,from_=1,to=1,textvariable=self.frame,width=6,command=self.frame_changed);self.frame_entry.pack(side='left',padx=3);self.frame_entry.bind('<Return>',lambda _:self.frame_changed())
        self.frame_label=ttk.Label(playback,text='/ 0');self.frame_label.pack(side='left')
        ttk.Label(playback,text='Target fps').pack(side='left',padx=(8,0));ttk.Spinbox(playback,from_=1,to=200,textvariable=self.fps,width=4).pack(side='left');ttk.Label(playback,textvariable=self.actual_fps).pack(side='left',padx=3)
        self.z_panel=ttk.Frame(playback)
        self.z_entry=ttk.Spinbox(self.z_panel,from_=1,to=1,textvariable=self.z,width=3,command=self.request_render);self.z_entry.pack(side='left',padx=3);ttk.Label(self.z_panel,text='Z').pack(side='left')
        self.z_entry.bind('<Return>',lambda _:self.request_render())
        curve_bar=ttk.Frame(curves,padding=(6,0));curve_bar.pack(fill='x')
        ttk.Label(curve_bar,text='Curve').pack(side='left');combo=ttk.Combobox(curve_bar,textvariable=self.curve_kind,values=['dF/F','Raw Intensity'],state='readonly',width=10);combo.pack(side='left',padx=5);combo.bind('<<ComboboxSelected>>',lambda _:self.plot_curves())
        self.curve_figure=Figure(figsize=(7,2),dpi=100,facecolor='white');self.curve_ax=self.curve_figure.add_subplot(111)
        self.curve_canvas=FigureCanvasTkAgg(self.curve_figure,master=curves);self.curve_blit=BlitCanvas(self.curve_canvas);synchronize_canvas(self.curve_canvas);self.curve_canvas.get_tk_widget().pack(fill='both',expand=True);self.cursor=None
        self.image_figure.text(.5,.5,'AQuA2\nLoad Image or Open Project',ha='center',va='center',fontsize=18,color='#7890a6');self.image_canvas.draw_idle()
    def _right(self,parent):
        body=self.section(parent,'Display')
        self.surface_panel=ttk.Frame(body)
        combo=ttk.Combobox(self.surface_panel,textvariable=self.volume_mode,values=['Gradient Opacity','3D Surface','Z Slice'],state='readonly');combo.pack(fill='x');combo.bind('<<ComboboxSelected>>',lambda _:self.change_volume())
        self.surface_options=ttk.Frame(self.surface_panel)
        ttk.Label(self.surface_options,text='Surface Level (fraction of peak)').pack(anchor='w')
        ttk.Scale(self.surface_options,from_=.05,to=.95,variable=self.surface_level,command=lambda _:self.request_render()).pack(fill='x')
        self.gradient_options=ttk.Frame(self.surface_panel)
        ttk.Label(self.gradient_options,text='Gradient Opacity').pack(anchor='w')
        ttk.Scale(self.gradient_options,from_=0,to=1,variable=self.gradient_opacity,command=lambda _:self.request_render()).pack(fill='x')
        row=ttk.Frame(self.surface_panel);row.pack(fill='x');ttk.Label(row,text='Z Scale').pack(side='left')
        scale=ttk.Spinbox(row,from_=.1,to=20,increment=.1,textvariable=self.z_scale,width=6,command=self.request_render);scale.pack(side='right');scale.bind('<Return>',lambda _:self.request_render())
        ttk.Label(self.surface_panel,text='Drag: rotate · Right-drag: pan\nWheel: zoom · Double-click: reset',wraplength=300).pack(anchor='w',pady=3)
        ttk.Checkbutton(body,text='All Events',variable=self.all_time,command=self.request_render).pack(anchor='w')
        for key,label in [('minimum','Min'),('maximum','Max'),('brightness','Brightness'),('contrast','Contrast'),('transparency','Transparency'),('overlay_brightness','Overlay Brightness')]:
            row=ttk.Frame(body)
            if key=='transparency':self.transparency_panel=row
            else:row.pack(fill='x')
            if key=='overlay_brightness':self.overlay_panel=row
            ttk.Label(row,text=label).pack(anchor='w')
            ttk.Scale(row,from_=0,to=1,variable=self.display_vars[key],command=lambda _,k=key:self.change_display(k)).pack(fill='x',pady=(0,5))
        row=ttk.Frame(body);row.pack(fill='x',pady=3);self.auto_button=ttk.Button(row,text='Auto',command=self.auto_display)
        self.auto_button.pack(side='left',expand=True,fill='x');self.reset_button=ttk.Button(row,text='Reset',command=self.reset_display)
        self.reset_button.pack(side='left',expand=True,fill='x',padx=(4,0))
        body=self.section(parent,'Events / CFUs')
        row=ttk.Frame(body);row.pack(fill='x');ttk.Entry(row,textvariable=self.search,width=12).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='Find ID',command=self.refresh_table).pack(side='left',padx=3)
        self.search.trace_add('write',lambda *_:self.refresh_table())
        row=ttk.Frame(body);row.pack(fill='x',pady=5);ttk.Button(row,text='☆ Favorite',command=self.favorite).pack(side='left',expand=True,fill='x');ttk.Button(row,text='Hide / Show',command=self.hide_selected).pack(side='left',expand=True,fill='x',padx=(4,0))
        table=ttk.Frame(parent);table.pack(fill='both',expand=True,pady=(0,5))
        self.tree=ttk.Treeview(table,columns=('favorite','id','start','area','metric'),show='headings',selectmode='extended',height=11)
        for key,text,width in [('favorite','☆',25),('id','ID',45),('start','Frame',55),('area','Size',65),('metric','dF/F',70)]:
            self.tree.heading(key,text=text,command=lambda c=key:self.sort_table(c));self.tree.column(key,width=width,minwidth=25,stretch=True,anchor='center')
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview);self.tree.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.tree.pack(fill='both',expand=True)
        self.tree.bind('<<TreeviewSelect>>',self.selection_changed)
        self.tree.bind('<Control-a>',lambda _:self.tree.selection_set(self.tree.get_children()) or 'break')
        self.count_label=ttk.Label(parent,text='0 items',padding=(8,2));self.count_label.pack(anchor='w')
        body=self.section(parent,'Groups / Details')
        self.group_combo=ttk.Combobox(body,textvariable=self.group_filter,state='readonly',values=['All']);self.group_combo.pack(fill='x');self.group_combo.bind('<<ComboboxSelected>>',self.select_group)
        self.details=tk.Text(body,height=7,width=34,wrap='word',relief='flat',background='white',font=('Microsoft YaHei UI',9));self.details.pack(fill='x',pady=(5,0));self.details.configure(state='disabled')
    def guarded(self):
        if self.busy or self.loading:self.status.set('Wait for the current task or cancel it.');return False
        return True
    def display_settings(self):return DisplaySettings(**{key:var.get() for key,var in self.display_vars.items()})
    def change_volume(self):
        self.pause_play();self.renderer.invalidate();self.display_key=None
        self.surface_options.pack_forget();self.gradient_options.pack_forget()
        if self.volume_mode.get()=='3D Surface':self.surface_options.pack(fill='x',after=self.surface_panel.winfo_children()[0])
        elif self.volume_mode.get()=='Gradient Opacity':self.gradient_options.pack(fill='x',after=self.surface_panel.winfo_children()[0])
        if self.navtoolbar.mode:
            if 'pan' in str(self.navtoolbar.mode).lower():self.navtoolbar.pan()
            else:self.navtoolbar.zoom()
        if self.volume_mode.get()=='Z Slice' and self.project and self.project.shape[2]>1:self.z_panel.pack(side='left')
        else:self.z_panel.pack_forget()
        if self.volume_mode.get()!='Z Slice' and self.gpu_canvas is None:
            self.use_gpu.set(True);self.toggle_gpu()
        self.request_render()
    def change_display(self,key):
        lo=self.display_vars['minimum'];hi=self.display_vars['maximum'];epsilon=1/65536
        if lo.get()>=hi.get():
            if key=='minimum':lo.set(min(1-epsilon,lo.get()));hi.set(min(1,lo.get()+epsilon))
            else:hi.set(max(epsilon,hi.get()));lo.set(max(0,hi.get()-epsilon))
        self.auto_divisor=0;self.display_revision+=1;self.request_render()
    def reset_display(self):
        for key,value in DisplaySettings().as_dict().items():self.display_vars[key].set(value)
        self.auto_divisor=0;self.auto_key=None;self.display_revision+=1;self.request_render()
    def auto_display(self):
        if not self.project or self.auto_pending or self.loading:return
        from .viewer_render import FrameRequest,frame_source,intensity_limits
        self.pause_play();project=self.project;t=self.frame.get()-1;z=self.z.get()-1;c=self.channel.get();view=self.view.get()
        if view.startswith('Rising Map'):return
        whole_volume=project.shape[2]>1 and self.volume_mode.get()!='Z Slice'
        key=(id(project),c,view,t,-1 if whole_volume else z)
        if key!=self.auto_key:self.auto_divisor=0
        self.auto_key=key;self.auto_divisor=5000 if self.auto_divisor<10 else self.auto_divisor//2
        divisor=self.auto_divisor;revision=self.display_revision
        request=FrameRequest(project,t,z,c,self.scope.get(),self.mode.get(),view,(),(),False,1.,.5,False,False,False,0)
        self.auto_pending=True;self.auto_button.configure(state='disabled')
        def calculate():
            try:
                if whole_volume:
                    from dataclasses import replace
                    image=np.stack([frame_source(replace(request,z=zz))[0] for zz in range(project.shape[2])],axis=2)
                else:image,_=frame_source(request)
                window=fiji_auto_range(image,divisor)
                limits=intensity_limits(project,c,'dF' if view=='dF / σ' else 'stored')
                self.ui_events.put(('auto_display',project,key,revision,window,limits,None))
            except Exception as error:self.ui_events.put(('auto_display',project,key,revision,None,None,str(error)))
        self.io.submit(calculate)
    def new_analysis(self):
        if self.guarded():self.preserve_changes(lambda:NewAnalysisDialog(self,self.start_job))
    def input_settings(self):
        return dict(parameters=str(self.parameter_path),preset=self.parameter_preset,gui_median=True,
                    options={**self.step_forms.get('preprocess'),**self.step_forms.get('detect')},cfu_settings=self.step_forms.get('cfu'))
    def load_parameters(self):
        if self.guarded():ParameterDialog(self,self.parameter_path,self.parameter_preset,self.apply_parameters)
    def apply_parameters(self,path,preset,options,settings):
        from .viewer_steps import PREPROCESS_FIELDS,DETECTION_FIELDS,validated
        validated({key:options.get(key,self.step_forms.defaults[key]) for key,*_ in PREPROCESS_FIELDS},PREPROCESS_FIELDS)
        validated({key:options.get(key,self.step_forms.defaults[key]) for key,*_ in DETECTION_FIELDS},DETECTION_FIELDS)
        self.step_forms.apply_parameters(options,settings);self.parameter_path=path;self.parameter_preset=preset
        if self.project:
            self.project.data['config']={**self.project.data.get('config',{}),'parameters':str(path),'preset':preset,
                'options':dict(options),'cfu_settings':settings};self.dirty=True
        self.status.set('Parameters loaded')
    def show_step(self,operation):
        if not self.project:return
        if operation=='preprocess' and 'dF' not in self.project.data['movies'][self.channel.get()]:self.status.set('Run preprocessing first.');return
        if operation=='detect' and 'detection' not in self.project.data['stages']:self.status.set('Run event detection first.');return
        if operation=='cfu' and 'cfu' not in self.project.data['stages']:self.status.set('Run CFU first.');return
        self.pause_play();self.mode.set('CFU' if operation=='cfu' else 'Event');self.scope.set('local')
        self.view.set({'preprocess':'dF / σ','detect':'Overlay','cfu':'Mean Projection'}[operation]);self.show_overlay.set(operation!='preprocess');self.changed()
    def step_job(self,operation):
        if not self.project:raise ValueError('Load an image first.')
        if operation=='detect':
            if not can_detect(self.project):raise ValueError('Run preprocessing first, including after reopening a saved project.')
            options=self.step_forms.get('preprocess')
            if any(not np.array_equal(value,self.project.opts.get(key)) for key,value in options.items()):raise ValueError('Parameters changed. Rerun preprocessing before detection.')
        if operation=='cfu' and 'detection' not in self.project.data['stages']:raise ValueError('Run event detection first.')
        return {'operation':operation,'settings' if operation=='cfu' else 'options':self.step_forms.get(operation)}
    def run_step(self,operation):
        if not self.guarded():return
        try:job=self.step_job(operation)
        except Exception as error:messagebox.showerror('Cannot Run',str(error),parent=self);return
        self.chain_steps.clear();self.chain_ready=False;self.start_job(job)
    def run_all(self):
        if not self.project or not self.guarded():return
        try:
            jobs=[{'operation':op,'settings' if op=='cfu' else 'options':self.step_forms.get(op)} for op in ('preprocess','detect','cfu')]
        except Exception as error:messagebox.showerror('Invalid Parameters',str(error),parent=self);return
        self.chain_steps=deque(jobs[1:]);self.chain_ready=False;self.start_job(jobs[0])
    def advance_chain(self):
        if self.chain_ready and self.chain_steps and not self.busy and not self.loading:
            self.chain_ready=False;self.start_job(self.chain_steps.popleft())
    def preserve_changes(self,action):
        if not self.dirty:action();return
        answer=messagebox.askyesnocancel('Unsaved Results','Save the current project before continuing? No continues without saving.',parent=self)
        if answer is True:self.save_to(self.project.path,action)
        elif answer is False:self.dirty=False;action()
    def open_dialog(self):
        if not self.guarded():return
        path=filedialog.askopenfilename(parent=self,filetypes=[('AQuA2 Python Project','*.h5')])
        if path:self.preserve_changes(lambda:self.open_project(path))
    def import_dialog(self):
        if not self.guarded():return
        if self.dirty:self.preserve_changes(self.import_dialog);return
        path=filedialog.askopenfilename(parent=self,title='Select AQuA2 MAT (res)',filetypes=[('AQuA2 MAT','*.mat')])
        if not path:return
        dialog=tk.Toplevel(self);dialog.title('Import MATLAB Results');dialog.transient(self)
        cfu=tk.StringVar()
        from .viewer_dialogs import file_entry
        ttk.Label(dialog,text='Keep existing events and features; link the original grayscale image.',padding=10).grid(row=0,column=0,columnspan=3,sticky='w')
        file_entry(dialog,'CFU file (optional)',cfu,1)
        ttk.Label(dialog,text='Results stay in memory. Save or export as needed.').grid(row=2,column=0,columnspan=3,padx=10,pady=8)
        def submit():
            job={'operation':'import','source':path}
            if cfu.get():job['cfu_file']=cfu.get()
            dialog.destroy();self.start_job(job)
        ttk.Button(dialog,text='Import ',command=submit,style='Primary.TButton').grid(row=3,column=1,pady=12)
        from .viewer_dialogs import center_dialog
        center_dialog(dialog,self);dialog.grab_set()
    def open_project(self,path):
        self.current_operation='open'
        self.pause_play();self.cancel_draw();self.renderer.invalidate()
        self.loading=True;self.status.set('Opening project…');self.begin_activity('Opening project…')
        def load():
            try:self.ui_events.put(('loaded',Project.open(path)))
            except Exception as error:self.ui_events.put(('load_error',str(error)))
        self.io.submit(load)
    def install_project(self,project,*,unsaved=False):
        self.pause_play();self.renderer.invalidate();self.cancel_draw();self.mask_revision+=1
        if self.current_operation in ('open','load_input','import'):
            for key,value in DisplaySettings().as_dict().items():self.display_vars[key].set(value)
        self.display_revision+=1
        self.project=project;self.loading=False;self.dirty=unsaved or project.path is None;self.selected=[];self.finish_activity('Project opened');self.display_shape=None;self.display_key=None;self.requested_frame=0;self.displayed_frame=0
        if project.shape[2]>1:
            self.surface_panel.pack(fill='x',before=self.surface_panel.master.winfo_children()[1]);self.volume_mode.set('Gradient Opacity' if self.use_gpu.get() else 'Z Slice')
            self.surface_options.pack_forget();self.gradient_options.pack(fill='x',after=self.surface_panel.winfo_children()[0])
        else:self.surface_panel.pack_forget();self.volume_mode.set('Z Slice')
        self.title('AQuA2 · '+project.data['title']);self.update_channels(project.channels)
        mode,view=display_for(project);self.mode.set(mode);self.view.set(view);self.scope.set('local');self.show_overlay.set(phase(project) in ('detection','cfu'));self.side.set(phase(project)!='input')
        self.search.set('');self.favorites_only.set(False);self.area_min.set('');self.duration_min.set('');self.dff_min.set('');self.group_filter.set('All')
        self.frame.set(1);self.slider.configure(to=project.shape[3]);self._updating_slider=True;self.slider.set(1);self._updating_slider=False;self.frame_entry.configure(to=project.shape[3]);self.z_entry.configure(to=project.shape[2]);self.z.set(1)
        if project.shape[2]>1 and self.volume_mode.get()=='Z Slice':self.z_panel.pack(side='left')
        else:self.z_panel.pack_forget()
        if project.shape[2]>1:self.transparency_panel.pack(fill='x',before=self.overlay_panel)
        else:self.transparency_panel.pack_forget()
        summary=project.summary();self.summary_text.set(f"{summary['title']}\n{project.shape[0]} × {project.shape[1]} × {project.shape[2]} · {project.shape[3]} frames\n"+' · '.join((f'CH2: ' if c=='channel2' else '')+f"{summary['events'][c]} events / {summary['cfus'][c]} CFUs" for c in project.channels))
        self.stage_list.configure(text={'input':'Image loaded','preprocess':'Completed: Preprocessing','detection':'Completed: Preprocessing → Event Detection','cfu':'Completed: Preprocessing → Event Detection → CFU'}[phase(project)])
        annotations=project.data['annotations'];self.region_text.configure(text=f"Region {len(annotations.get('regions',[]))} · Landmark {len(annotations.get('landmarks',[]))}"+(' (Pending)' if annotations.get('regions_dirty') else ''))
        self.step_forms.load(project,self.current_operation);self.run_all_button.configure(state='normal');self.update_views()
        self.update_groups();self.refresh_table();self.plot_curves();self.request_render()
        if phase(project)=='detection' and self.rows:
            target=self.rows[0].get('peak',self.rows[0]['start'])
            self.set_frame(int(target if np.isfinite(target) else self.rows[0]['start']))
        stage=phase(project);label={'input':'Image loaded. Ready for preprocessing.','preprocess':'Preprocessing complete','detection':'Event detection complete','cfu':'CFU complete'}[stage]
        self.task_text.set(label+(' (Unsaved)' if self.dirty else ''));self.status.set(label+'. Adjust parameters, run the next stage or save.')
    def update_channels(self,channels):
        self.channel_combo.configure(values=channels)
        if self.channel.get() not in channels:self.channel.set(channels[0])
        if 'channel2' in channels:self.channel_combo.pack(side='left',padx=2,before=self.scope_combo)
        else:self.channel_combo.pack_forget()
    def changed(self,*args):
        self.update_views()
        self.selected=[];self.group_filter.set('All');self.update_groups();self.refresh_table();self.plot_curves();self.request_render()
    def update_views(self):
        if not self.project:return
        values=['Overlay','Original','Mean Projection','Max Projection']
        if 'dF' in self.project.data['movies'][self.channel.get()]:values.append('dF / σ')
        if 'detection' in self.project.data['stages'] and self.mode.get()=='Event':values+=['Rising Map 20%','Rising Map 50%','Rising Map 80%']
        self.view_combo.configure(values=values)
        if self.view.get() not in values:self.view.set('Mean Projection' if self.mode.get()=='CFU' else 'Original')
    def favorites_key(self):return self.mode.get()+':'+self.channel.get()+':'+('local' if self.mode.get()=='CFU' else self.scope.get())
    def update_groups(self):
        if not self.project:return
        groups=self.project.data['stages'].get('cfu',{}).get('cfuGroupInfo',np.empty((0,4),object))
        self.group_combo.configure(values=['All']+[f'Group {i+1} · {np.asarray(row[1]).size} CFU' for i,row in enumerate(groups)])
    def refresh_table(self):
        if not self.project:return
        selection={int(v) for v in self.tree.selection()};self.tree.delete(*self.tree.get_children());channel=self.channel.get();scope=self.scope.get()
        if self.mode.get()=='CFU':
            self.tree.heading('start',text='#Event');self.tree.heading('area',text='FrequencyHz');self.tree.heading('metric',text='Peak dF/F')
            self.rows=[{'id':i,'start':len(r['eventIds']),'area':r['frequencyStats'].get('mainFreq',np.nan) if isinstance(r['frequencyStats'],dict) else np.nan,'dff':float(np.nanmax(r['meanDff'])) if len(r['meanDff']) else np.nan,'duration':np.nan} for i,r in enumerate(self.project.cfus(channel))]
        else:
            self.tree.heading('start',text='Start');self.tree.heading('area',text='Size');self.tree.heading('metric',text='dF/F');self.rows=event_rows(self.project,channel,scope)
        favorite=favorites(self.project,channel,scope,self.mode.get());hidden=set(self.project.data['annotations'].get('hidden',{}).get(self.favorites_key(),[]))
        try:
            thresholds=[float(v.get()) if v.get().strip() else None for v in (self.area_min,self.duration_min,self.dff_min)]
            query={int(v)-1 for v in self.search.get().replace(', ',',').replace(' ', ',').split(',') if v.strip()}
        except ValueError:self.status.set('Enter numeric filters; separate IDs with commas.');return
        self.visible=[]
        for row in self.rows:
            i=row['id']
            if query and i not in query:continue
            if self.favorites_only.get() and i not in favorite:continue
            if self.mode.get()!='CFU' and any(limit is not None and (not np.isfinite(row[key]) or row[key]<limit) for key,limit in zip(('area','duration','dff'),thresholds)):continue
            values=('★' if i in favorite else ('◌' if i in hidden else ''),i+1,_number(row['start']),_number(row['area']),_number(row['dff']))
            self.tree.insert('',tk.END,iid=str(i),values=values)
            if i not in hidden:self.visible.append(i)
        available=set(self.tree.get_children());keep=[str(i) for i in selection if str(i) in available]
        if keep:self.tree.selection_set(keep)
        self.count_label.configure(text=f'{len(available)} / {len(self.rows)} items · {len(favorite)} Favorites')
        self.request_render()
    def sort_table(self,column):
        children=list(self.tree.get_children())
        def key(i):
            value=self.tree.set(i,column)
            try:return (0,float(value))
            except ValueError:return (1,value)
        for position,i in enumerate(sorted(children,key=key)):self.tree.move(i,'',position)
    def selection_changed(self,*args):
        self.selected=[int(i) for i in self.tree.selection()]
        if self.selected and self.mode.get()!='CFU':
            i=self.selected[0]
            if i<len(self.rows):self.set_frame(int(self.rows[i].get('peak',self.rows[i]['start'])))
        self.show_details();self.plot_curves();self.request_render()
    def show_details(self):
        text='Select an event or CFU for details.'
        if self.project and self.selected:
            i=self.selected[0]
            if self.mode.get()=='CFU':
                record=self.project.cfus(self.channel.get())[i];ids=', '.join(str(e+1) for e in record['eventIds'])
                text=f"CFU {i+1}\n{len(record['eventIds'])} shared/local events\nEvent ID: {ids}\nFrequency: {_number(record['frequencyStats'].get('mainFreq',np.nan)) if isinstance(record['frequencyStats'],dict) else '—'} Hz"
            else:
                row=self.rows[i];unit='μm³' if self.project.shape[2]>1 else 'μm²';text=f"Event {i+1}\nFrames {int(row['start'])}–{int(row['end'])}\nSize: {_number(row['area'])} {unit}\nDuration: {_number(row['duration'])} s\nPeak dF/F: {_number(row['dff'])}\nDecay τ: {_number(row['tau'])} s\nVoxels: {row['voxels']}"
        self.details.configure(state='normal');self.details.delete('1.0','end');self.details.insert('1.0',text);self.details.configure(state='disabled')
    def select_group(self,*args):
        if not self.project or self.group_filter.get()=='All':return
        self.mode.set('CFU');self.search.set('');self.favorites_only.set(False);self.refresh_table();group=int(self.group_filter.get().split()[1])-1
        rows=self.project.data['stages']['cfu']['cfuGroupInfo'];ids=np.asarray(rows[group,1],np.int64).ravel()-1
        offset=0 if self.channel.get()=='channel1' else len(self.project.cfus('channel1'))
        available=set(self.tree.get_children());ids=[str(i-offset) for i in ids if str(i-offset) in available];self.tree.selection_set(ids)
        if ids:self.tree.see(ids[0])
    def favorite(self):
        if not self.project or not self.guarded():return
        key=self.favorites_key();values=favorites(self.project,self.channel.get(),self.scope.get(),self.mode.get())
        stored=self.project.data['annotations'].setdefault('favorites',{})
        if self.mode.get()=='CFU':stored.pop('CFU:'+self.channel.get()+':global',None)
        else:
            for alias in (self.channel.get()+':'+self.scope.get(),'Event:'+self.channel.get()+':'+self.scope.get()):stored.pop(alias,None)
        for i in self.selected:
            if i in values:values.remove(i)
            else:values.add(i)
        self.project.data['annotations']['favorites'][key]=sorted(values);self.dirty=True;self.refresh_table()
    def hide_selected(self):
        if not self.project or not self.guarded():return
        key=self.favorites_key();values=set(self.project.data['annotations'].setdefault('hidden',{}).get(key,[]))
        for i in self.selected:
            if i in values:values.remove(i)
            else:values.add(i)
        self.project.data['annotations']['hidden'][key]=sorted(values);self.dirty=True;self.refresh_table()
    def set_frame(self,value,*,playback=False):
        if not self.project or self.draw_kind:return
        if not playback:self.pause_play()
        self.requested_frame=max(0,min(self.project.shape[3]-1,int(value)-1));self.request_render()
    def slide(self,value):
        if self.project and not self._updating_slider:self.set_frame(int(round(float(value))))
    def frame_changed(self):
        try:self.set_frame(self.frame.get())
        except tk.TclError:self.status.set('Frame must be an integer.')
    def step(self,delta):
        if self.project:self.set_frame(self.requested_frame+1+delta)
    def space_key(self,event):
        # Buttons already invoke on Space release. A toplevel toggle as well
        # would start and stop playback in the same keystroke.
        if not isinstance(self.focus_get(),(ttk.Entry,ttk.Spinbox,tk.Text,ttk.Button,tk.Button,ttk.Checkbutton,ttk.Radiobutton,ttk.Combobox)):
            self.toggle_play();return 'break'
    def pause_play(self):
        was_playing=self.playing
        self.playing=False;self.play_deadline=None
        self.ready_render=None;self.present_times.clear();self.actual_fps.set('')
        if self.play_after is not None:self.after_cancel(self.play_after);self.play_after=None
        self.play_button.configure(text='▶ Play')
        if was_playing:self.renderer.invalidate();self.requested_frame=self.displayed_frame
    def toggle_play(self):
        if not self.project or self.draw_kind or self.loading:return
        if self.playing:self.pause_play()
        else:
            self.playing=True;self.play_deadline=time.perf_counter();self.play_button.configure(text='Ⅱ Pause');self.play_tick()
    def play_tick(self):
        self.play_after=None
        if not self.playing or not self.project or self.draw_kind:return
        self.set_frame((self.displayed_frame+1)%self.project.shape[3]+1,playback=True)
    def request_render(self):
        if not self.project or self.draw_kind or getattr(self,'loading',False):return
        try:
            t=self.requested_frame;z=self.z.get()-1;z_scale=max(.1,min(20,float(self.z_scale.get())))
        except (tk.TclError,ValueError):self.status.set('Enter a numeric Z scale.');return
        if not 0<=t<self.project.shape[3] or not 0<=z<self.project.shape[2]:return
        if self.play_after is not None:self.after_cancel(self.play_after);self.play_after=None
        self.renderer.submit(FrameRequest(self.project,t,z,self.channel.get(),self.scope.get(),self.mode.get(),self.view.get(),
            tuple(self.selected),tuple(self.visible),self.all_time.get(),1.,.5,self.show_overlay.get(),self.side.get(),self.y_up.get(),self.mask_revision,False,annotate=True,label_font=self.label_font_size(),display=self.display_settings(),
            volume=self.project.shape[2]>1 and self.volume_mode.get()!='Z Slice' and not self.view.get().startswith('Rising Map'),surface_level=float(self.surface_level.get()),z_scale=z_scale,
            volume_style='gradient' if self.volume_mode.get()=='Gradient Opacity' else 'surface',gradient_opacity=float(self.gradient_opacity.get())))
        self.ready_render=None
    def label_font_size(self):
        if getattr(self,'image_axes',None):
            ax=self.image_axes[0];pixels_per_source=ax.bbox.width/max(abs(np.diff(ax.get_xlim())[0]),1)
        else:pixels_per_source=self.image_figure.bbox.width/(2 if self.side.get() else 1)/self.project.shape[1]
        return max(6,min(48,round(12*self.image_canvas.device_pixel_ratio/max(pixels_per_source,.1))))
    def poll_render(self):
        result=self.renderer.poll()
        if result is not None:self.ready_render=result
        result=getattr(self,'ready_render',None)
        if result is not None and result[0]!=getattr(self.renderer,'generation',result[0]):self.ready_render=None;result=None
        if result is not None and self.playing and time.perf_counter()<(self.play_deadline or 0):result=None
        if result is not None:
            self.ready_render=None
            token,request,arrays,error,started=result
            if error is not None:
                if hasattr(self,'chain_steps'):self.chain_steps.clear();self.chain_ready=False
                self.pause_play();self.status.set(str(error))
            elif request.project is self.project and not self.draw_kind:
                self.draw_image(request,*arrays)
                # Commit controls and curve cursor only after this image was
                # actually painted. Slow storage cannot advance the playhead.
                self.displayed_frame=request.t;self.frame.set(request.t+1)
                self._updating_slider=True;self.slider.set(request.t+1);self._updating_slider=False
                seconds=request.t*float(request.project.opts['frameRate'])
                self.frame_label.configure(text=f'/ {request.project.shape[3]} · {seconds:.2f}s')
                if self.cursor is not None:
                    self.cursor.set_xdata([seconds]*2);self.curve_blit.paint_cursor(self.cursor)
                if getattr(self,'chain_ready',False):self.after_idle(self.advance_chain)
                if self.playing:
                    now=time.perf_counter();self.present_times.append(now)
                    if len(self.present_times)>1 and now-self.rate_updated>.25:
                        self.actual_fps.set(f'Actual {(len(self.present_times)-1)/(now-self.present_times[0]):.1f}');self.rate_updated=now
                    try:fps=max(1,int(self.fps.get()))
                    except tk.TclError:fps=10
                    self.play_deadline,delay=playback_delay(self.play_deadline if self.play_deadline is not None else started,time.perf_counter(),fps)
                    # Prepare the next frame immediately while the current
                    # one is on screen. Hold it until its presentation deadline.
                    self.play_tick()
        self.after(1 if self.playing or getattr(self.renderer,'active',False) else 16,self.poll_render)
    def draw_image(self,request,image,raw,rgba,hits):
        from .viewer_volume import VolumeScene
        if isinstance(image,VolumeScene):
            if self.gpu_canvas is not None:
                try:self.gpu_canvas.present_volume(image);self.display_key=('volume',request.side);self.hits=None;return
                except Exception as error:self.gpu_canvas.fail(error)
            self.volume_mode.set('Z Slice');self.status.set('3D GPU unavailable. Showing Z slices.');self.request_render();return
        from .viewer_surface import SurfaceScene
        if isinstance(image,SurfaceScene):
            if self.gpu_canvas is not None:
                try:
                    self.gpu_canvas.present_surface(image);self.display_key=('surface',request.side);self.hits=None;return
                except Exception as error:self.gpu_canvas.fail(error)
            self.volume_mode.set('Z Slice');self.status.set('3D GPU unavailable. Showing Z slices.');self.request_render();return
        view=request.view;self.hits=hits;count=2 if request.side else 1;rising=view.startswith('Rising Map')
        key=(id(request.project),image.shape,count,request.y_up,view,request.z,request.mask_revision)
        if self.display_key!=key:
            limits=None
            if getattr(self,'display_shape',None)==image.shape and getattr(self,'image_axes',None):
                old=self.image_axes[0];xlim=old.get_xlim();ylim=sorted(old.get_ylim())
                if not (np.allclose(xlim,(-.5,image.shape[1]-.5)) and np.allclose(ylim,(-.5,image.shape[0]-.5))):limits=(xlim,ylim)
            self.display_shape=image.shape;self.image_blit.set_artists([]);self.image_figure.clear()
            axes=self.image_figure.subplots(1,count,squeeze=False,sharex=True,sharey=True)[0]
            axes=list(reversed(axes)) if request.side else list(axes);self.image_axes=axes
            origin='lower' if request.y_up else 'upper';self.image_artists=[];dynamic=[];self.rising_bar=None
            for i,ax in enumerate(axes):
                ax.set_facecolor('#101820');ax.set_xticks([]);ax.set_yticks([])
                if i==0 and rising:
                    artist=ax.imshow(image,origin=origin,cmap=matplotlib.colormaps['turbo'].with_extremes(bad='#101820'),interpolation='nearest')
                    self.rising_bar=self.image_figure.colorbar(artist,ax=ax,fraction=.045,pad=.02);self.rising_bar.set_label('Rise Time (s)',fontsize=8)
                else:artist=pixel_imshow(ax,image if i==0 else raw,origin=origin,cmap=matplotlib.colormaps['gray'].with_extremes(bad='#101820'),vmin=0,vmax=1,interpolation='nearest')
                self.image_artists.append(artist);dynamic.append(artist)
                if i==0:
                    self.overlay_artist=ax.imshow(rgba if rgba is not None else np.zeros((1,1,4)),origin=origin,interpolation='nearest',extent=(-.5,image.shape[1]-.5,image.shape[0]-.5,-.5) if origin=='upper' else (-.5,image.shape[1]-.5,-.5,image.shape[0]-.5));dynamic.append(self.overlay_artist)
                    dynamic.extend(self.draw_masks(ax,request.z))
                ax.set_title(view if i==0 else 'Original',fontsize=10,color='#314b64')
                if limits:ax.set_xlim(limits[0]);ax.set_ylim(limits[1] if request.y_up else list(reversed(limits[1])))
            self.image_figure.subplots_adjust(left=.015,right=.985,bottom=.02,top=.94,wspace=.28 if rising and count==2 else .06)
            self.image_blit.set_artists(dynamic);self.display_key=key
        else:
            self.image_artists[0].set_data(image)
            if request.side:self.image_artists[1].set_data(raw)
            if rising:
                finite=image[np.isfinite(image)]
                if finite.size:
                    lo,hi=float(finite.min()),float(finite.max())
                    if hi<=lo:hi=lo+1
                    if self.image_artists[0].get_clim()!=(lo,hi):
                        self.image_artists[0].set_clim(lo,hi);self.image_blit.invalidate()
        if rgba is not None:self.overlay_artist.set_data(rgba)
        self.overlay_artist.set_visible(rgba is not None and image.ndim!=3 and request.overlay and view!='Original' and not rising)
        gpu=getattr(self,'gpu_canvas',None)
        if gpu is not None and image.dtype==np.uint8 and image.ndim==3 and gpu.supported(request):
            try:gpu.present(request,image,raw);return
            except Exception as error:
                gpu.close();self.gpu_canvas=None;self.use_gpu.set(False);self.status.set('GPU failed; using fallback: '+str(error))
        elif gpu is not None:gpu.window.hide()
        self.image_blit.paint()
    def draw_masks(self,ax,z=None):
        artists=[]
        if not self.project:return artists
        if z is None:z=self.z.get()-1
        for key,stroke in (('regions','#43d9e6'),('landmarks','#ffcf56')):
            for pixels in self.project.data['annotations'].get(key,[]):
                p=np.asarray(pixels,np.int64);h,w,depth,_=self.project.shape;p=p[p//(h*w)==z]%(h*w)
                if not len(p):continue
                mask=np.zeros((h,w));mask[p%h,p//h]=1;artists.append(ax.contour(mask,levels=[.5],colors=[stroke],linewidths=.9))
        return artists
    def plot_curves(self):
        ax=self.curve_ax;ax.clear();ax.set_facecolor('white');ax.grid(alpha=.16);ax.set_xlabel('Time (s)',fontsize=8);ax.set_ylabel(self.curve_kind.get(),fontsize=8);ax.tick_params(labelsize=8)
        if self.project:
            series=curve_series(self.project,self.mode.get(),self.channel.get(),self.scope.get(),self.selected[:8],self.curve_kind.get(),False)
            infos=self.project.cfus(self.channel.get()) if self.mode.get()=='CFU' else None
            for index,x,y,occurrences in series:
                plot_curve(ax,index,x,y,record=infos[index] if infos is not None else None,label=f'{self.mode.get()} {index+1}')
                if len(occurrences):ax.plot(occurrences,np.full(len(occurrences),np.nanmin(y)),linestyle='none',marker='|',color=color(index),alpha=.65)
            if series and infos is not None:ax.set_title('Bold: member events　Gray: other events　Thin: inactive',fontsize=8,loc='left')
            if series:ax.legend(fontsize=7,loc='upper right',ncol=min(4,len(series)))
            else:ax.text(.5,.5,'Curves are not available in this result.' if self.selected else 'Click an image or list item to view curves.',ha='center',va='center',transform=ax.transAxes,color='#8b9caf',fontsize=9)
            self.cursor=ax.axvline((self.frame.get()-1)*self.project.opts['frameRate'],color='#51677d',linewidth=.8,linestyle='--')
        self.curve_figure.subplots_adjust(left=.09,right=.98,bottom=.24,top=.92);self.curve_blit.set_artists([self.cursor] if self.cursor is not None else []);self.curve_canvas.draw_idle()
    def pick_pixel(self,event):
        if self.draw_kind or self.hits is None or event.xdata is None or event.ydata is None:return
        row,column=int(round(event.ydata)),int(round(event.xdata))
        if not 0<=row<self.hits.shape[0] or not 0<=column<self.hits.shape[1]:return
        value=int(self.hits[row,column])-1
        if value>=0 and self.tree.exists(str(value)):self.tree.selection_set(str(value));self.tree.see(str(value))
    def draw_region(self,kind):
        if hasattr(self,'volume_mode') and self.volume_mode.get()!='Z Slice':
            self.volume_mode.set('Z Slice');self.change_volume();self.status.set('Switched to Z Slice. Draw on the current slice.');return
        if self.project and phase(self.project)=='input':self.status.set('Run preprocessing before drawing regions on the cropped image.');return
        if not self.project or not self.guarded() or not getattr(self,'image_axes',None):return
        self.pause_play();self.cancel_draw();self.renderer.invalidate();self.draw_kind=kind;self.requested_frame=self.displayed_frame
        for widget in (self.z_entry,self.frame_entry,self.slider):widget.configure(state='disabled')
        self.status.set('Click vertices on the left image; click the first vertex to close. Esc cancels.')
        if self.navtoolbar.mode:
            if 'pan' in str(self.navtoolbar.mode).lower():self.navtoolbar.pan()
            else:self.navtoolbar.zoom()
        self.image_blit.suspend()
        def finish(vertices):
            try:pixels=polygon_pixels(vertices,self.project.shape,self.z.get()-1)
            except ValueError as error:
                self.cancel_draw();self.status.set(str(error));return
            if not len(pixels):self.cancel_draw();self.status.set('No pixels inside the polygon. Mask not added.');return
            annotations=self.project.data['annotations'];annotations.setdefault(kind,[]).append(pixels);annotations['regions_dirty']=True;self.dirty=True;self.mask_revision+=1
            name_key='region_names' if kind=='regions' else 'landmark_names'
            if name_key in annotations:annotations[name_key].append(str(len(annotations[kind])))
            self.region_text.configure(text=f"Region {len(annotations['regions'])} · Landmark {len(annotations['landmarks'])}(Pending)")
            self.cancel_draw();self.status.set('Polygon complete');self.request_render()
        self.polygon=PolygonSelector(self.image_axes[1] if len(self.image_axes)==2 else self.image_axes[0],finish,useblit=True,grab_range=10,props={'color':'#ffcf56' if kind=='landmarks' else '#43d9e6','linewidth':1.5})
        self.bind('<Escape>',lambda _:self.cancel_draw())
    def cancel_draw(self):
        if self.polygon:
            self.polygon.set_active(False);self.polygon.disconnect_events()
            for artist in self.polygon.artists:artist.remove()
            self.polygon=None
        drawing=self.draw_kind is not None;self.draw_kind=None
        for widget in (self.z_entry,self.frame_entry,self.slider):widget.configure(state='normal')
        if drawing:
            self.image_blit.resume();self.status.set('Drawing canceled. Existing masks retained.')
    def undo_mask(self,kind):
        if not self.project or not self.guarded():return
        annotations=self.project.data['annotations'];values=annotations.get(kind,[])
        if not values:return
        values.pop();self.mask_revision+=1;name_key='region_names' if kind=='regions' else 'landmark_names'
        if annotations.get(name_key):annotations[name_key].pop()
        annotations['regions_dirty']=True;self.dirty=True
        self.region_text.configure(text=f"Region {len(annotations['regions'])} · Landmark {len(annotations['landmarks'])}(Pending)");self.request_render()
    def import_masks(self,kind):
        if not self.project or not self.guarded():return
        label='Region' if kind=='regions' else 'Landmark'
        path=filedialog.askopenfilename(parent=self,title='Import '+label+' Mask',filetypes=[('Mask MAT','*.mat')])
        if not path:return
        try:
            from .features import masks_from_mat
            from .fea_region import mask_pixels
            values=masks_from_mat(path,kind);pixels=mask_pixels(values,self.project.shape[:3])
            self.project.data['annotations'][kind]=pixels
            name_key='region_names' if kind=='regions' else 'landmark_names'
            self.project.data['annotations'][name_key]=[str(i+1) for i in range(len(pixels))]
            if phase(self.project)=='input':
                config=self.project.data.setdefault('config',{});config[kind]=path
                config.setdefault('preview_mask_coordinates',{})[kind]=tuple(self.project.shape[:3])
            self.project.data['annotations']['regions_dirty']=True;self.dirty=True;self.mask_revision+=1
            self.region_text.configure(text=f"Region {len(self.project.data['annotations']['regions'])} · Landmark {len(self.project.data['annotations']['landmarks'])}(Pending)");self.request_render()
        except Exception as error:messagebox.showerror('Mask Import Failed',str(error),parent=self)
    def save(self):
        if not self.project or not self.guarded():return
        self.save_to(self.project.path)
    def save_as(self):
        if not self.project or not self.guarded():return
        path=filedialog.asksaveasfilename(parent=self,defaultextension='.aqua.h5',filetypes=[('AQuA2 Python Project','*.h5')])
        if path:self.save_to(path)
    def save_to(self,path,after=None):
        if path is None:
            path=filedialog.asksaveasfilename(parent=self,defaultextension='.aqua.h5',filetypes=[('AQuA2 Python Project','*.h5')])
            if not path:return
        self.pause_play();self.cancel_draw();released=self.renderer.release();self.loading=True;self.status.set('Saving project…');self.begin_activity('Saving project…');project=self.project
        def save():
            try:released.result();self.ui_events.put(('saved',project.save(path),after))
            except Exception as error:self.ui_events.put(('load_error',str(error)))
        self.io.submit(save)
    def start_job(self,job):
        if not self.guarded():return
        job=dict(job)
        if self.project and job['operation']=='cfu':job['feature_options']={'networkFeatures':self.step_forms.flags['networkFeatures'].get()}
        if self.project and job['operation'] in ('preprocess','detect','cfu','features','dependencies','export','relink'):
            job.pop('project',None);job['project_data']=preprocessing_input(self.project) if job['operation']=='preprocess' else self.project.data;job['project_path']=str(self.project.path) if self.project.path else None
        self.current_operation=job['operation']
        self.completed_analysis=None;self.exported_directory=None
        self.pause_play();self.cancel_draw();self.renderer.invalidate();self.busy=True;self.completed_path=None;self.last_error=None;self.status.set('Starting task…');self.begin_activity('Starting task…');self.cancel_button.configure(state='normal')
        self.step_forms.update(busy=True);self.run_all_button.configure(state='disabled')
        try:self.job.start(job)
        except Exception as error:
            self.busy=False;self.chain_steps.clear();self.chain_ready=False;self.step_forms.update();self.run_all_button.configure(state='normal' if self.project else 'disabled')
            self.finish_activity('Task Failed to Start');self.cancel_button.configure(state='disabled');messagebox.showerror('Task Failed to Start',str(error),parent=self)
    def run_cfu_dialog(self):
        if not self.project or not self.guarded():return
        settings=self.project.data['stages'].get('cfu',{}).get('cfuOpts',default_options())
        CFUSettingsDialog(self,settings,lambda value:self.start_job({'operation':'cfu','project':str(self.project.path),'settings':value}),dual_channel='channel2' in self.project.channels)
    def relink_dialog(self):
        if not self.project or not self.guarded():return
        path=filedialog.askopenfilename(parent=self,title='Relink Original MAT',filetypes=[('Saved MAT','*.mat')])
        if path:self.start_job({'operation':'relink','project':str(self.project.path),'source':path})
    def run_features(self):
        if self.project:self.start_job({'operation':'features','project':str(self.project.path)})
    def export_dialog(self):
        if not self.project or not self.guarded():return
        selection={key:var.get() for key,var in self.export_choices.items()}
        if not any(selection.values()):messagebox.showinfo('Export Selection','Select at least one export item.',parent=self);return
        directory=filedialog.askdirectory(parent=self,title='Export Results To')
        if directory:self.start_job({'operation':'export','directory':directory,'selection':selection,
            'display_settings':dict(display=self.display_settings().as_dict(),y_up=self.y_up.get(),annotate=True,label_font=self.label_font_size(),
                mode=self.mode.get(),channel=self.channel.get(),scope=self.scope.get(),view=self.view.get(),selected=list(self.selected),all_time=self.all_time.get())})
    def export_figure(self):
        if not self.project:return
        if self.project.shape[2]>1 and self.volume_mode.get()!='Z Slice' and self.gpu_canvas:
            path=filedialog.asksaveasfilename(parent=self,defaultextension='.png',filetypes=[('PNG Image','*.png')])
            if path:
                try:
                    from PIL import Image
                    Image.fromarray(self.gpu_canvas.redraw(capture=True)).save(path);self.status.set('View exported: '+path)
                except Exception as error:messagebox.showerror('Export Failed',str(error),parent=self)
            return
        path=filedialog.asksaveasfilename(parent=self,defaultextension='.png',filetypes=[('PNGImage','*.png'),('SVGVector','*.svg')])
        if path:
            try:self.image_figure.savefig(path,dpi=200,bbox_inches='tight');self.status.set('View exported: '+path)
            except Exception as error:messagebox.showerror('Export Failed',str(error),parent=self)
    def cancel(self):
        self.chain_steps.clear();self.chain_ready=False
        self.job.cancel();self.status.set('Canceling. Current project retained.');self.task_text.set('Canceling task…')
    def show_log(self):
        window=tk.Toplevel(self);window.title('AQuA2 Task Log');text=tk.Text(window,width=110,height=32,wrap='word');text.pack(fill='both',expand=True)
        if self.job.log_path and self.job.log_path.exists():text.insert('1.0',self.job.log_path.read_text(encoding='utf-8'))
        else:text.insert('1.0','No task log yet.')
        text.configure(state='disabled')
        from .viewer_dialogs import center_dialog
        center_dialog(window,self)
    def help(self):
        messagebox.showinfo('AQuA2 Help',
            '1. Load Parameters applies a CSV preset to all three stages.\n'
            '2. Load Image reads TIFF, MAT or AVI and detects channels. Use Import MAT for saved results and select the matching CFU file.\n'
            '3. Run Preprocessing, Event Detection and CFU in order, or use Run All. Additional parameters are under Advanced.\n'
            '4. Original image on the left, results on the right. Click an image or list item for curves. Arrows step; Space plays. Use Z for 3D slices.\n'
            '5. Draw polygons on the left image; click the first vertex to close. Esc cancels. Runs use current regions and landmarks.\n'
            '6. Use Save or Save As to save a project. Reopened projects require preprocessing before detection. Export tables, masked images and rising maps as needed.\n'
            '7. Gradient Opacity: drag to rotate, right-drag to pan, wheel to zoom, double-click to reset. Gradient Opacity controls uniform-region transparency. Export View saves the current camera.',parent=self)
    def poll(self):
        try:
            while True:
                message=self.ui_events.get_nowait();kind=message[0]
                if kind=='loaded':self.install_project(message[1])
                elif kind=='auto_display':
                    self.auto_pending=False;self.auto_button.configure(state='normal')
                    _,project,key,revision,window,limits,error=message
                    whole_volume=self.project and self.project.shape[2]>1 and hasattr(self,'volume_mode') and self.volume_mode.get()!='Z Slice'
                    current=(id(self.project),self.channel.get(),self.view.get(),self.frame.get()-1,-1 if whole_volume else self.z.get()-1)
                    if current!=key or self.display_revision!=revision:continue
                    if error:self.status.set('Auto failed: '+error);continue
                    minimum,maximum=0.,1.
                    if window is not None:
                        base,top=limits;span=max(top-base,1e-12)
                        minimum=float(np.clip((window[0]-base)/span,0,1-1/65536))
                        maximum=float(np.clip((window[1]-base)/span,minimum+1/65536,1))
                    self.display_vars['minimum'].set(minimum);self.display_vars['maximum'].set(maximum)
                    self.display_vars['brightness'].set(.5);self.display_vars['contrast'].set(.5)
                    self.display_revision+=1;self.request_render()
                elif kind=='saved':
                    # Keep the unquantized in-memory stage across manual
                    # saving; it is deliberately absent from the disk file.
                    if self.project and can_detect(self.project):
                        message[1].data['runtime']=self.project.data['runtime']
                        for channel in self.project.channels:message[1].data['movies'][channel]['working']=self.project.data['movies'][channel]['working']
                    self.project=message[1];self.loading=False;self.dirty=False;self.finish_activity('Project saved');self.status.set('Project saved.')
                    self.step_forms.project=self.project;self.step_forms.update()
                    if message[2]:message[2]()
                elif kind=='load_error':self.loading=False;self.finish_activity('Project Operation Failed');self.status.set(message[1]);messagebox.showerror('Project Operation Failed',message[1],parent=self)
        except queue.Empty:pass
        try:
            while True:
                message=self.job.events.get_nowait();kind=message.get('type')
                if kind=='progress':
                    self.update_task_progress(message)
                elif kind=='heartbeat':self.progress_state.cpu_seconds=message.get('cpu_seconds')
                elif kind=='complete':self.completed_analysis=message.get('analysis');self.exported_directory=message.get('exported')
                elif kind=='error':self.last_error=_job_error(message['error']);self.status.set('Task failed: '+self.last_error)
                elif kind=='exit':
                    self.busy=False;self.finish_activity('Task finished');self.cancel_button.configure(state='disabled')
                    self.step_forms.update();self.run_all_button.configure(state='normal' if self.project else 'disabled')
                    if message['cancelled']:
                        self.chain_steps.clear();self.chain_ready=False;self.status.set('Task canceled. Previous results retained.');self.task_text.set('Task canceled')
                    elif message['code']==0 and self.completed_analysis is not None:
                        self.install_project(self.completed_analysis,unsaved=True);self.chain_ready=bool(self.chain_steps)
                    elif message['code']==0 and self.exported_directory:self.status.set('Results exported: '+self.exported_directory);self.task_text.set('Export complete')
                    else:
                        self.chain_steps.clear();self.chain_ready=False
                        error=self.last_error or 'Worker failed. See Task Log for details.'
                        self.status.set('Task failed: '+error);self.task_text.set('Task failed: '+error);messagebox.showerror('Task Failed',error,parent=self)
        except queue.Empty:pass
        if self.busy or self.loading:self.task_timing.set(self.progress_state.timing())
        self.after(100,self.poll)
    def close(self):
        if self.loading:
            messagebox.showinfo('Project Operation in Progress','Wait for the project operation to finish before exiting.',parent=self)
            return
        message='Exit AQuA2?'
        if self.busy:message+='\n\nExiting cancels the current task. Incomplete results will not be saved.'
        if self.dirty:message+='\n\nUnsaved results, favorites and region/landmark edits will be discarded.'
        if not messagebox.askokcancel('Confirm Exit',message,parent=self,default=messagebox.CANCEL,icon=messagebox.QUESTION):return
        self.playing=False;self.job.cancel();self.io.shutdown(wait=False,cancel_futures=True)
        if getattr(self,'gpu_canvas',None) is not None:self.gpu_canvas.close()
        if getattr(self,'renderer',None):self.renderer.close()
        self.destroy()


def launch(project=None,*,image_backend='auto'):
    Workbench(project,image_backend=image_backend).mainloop()
