"""Native file/settings dialogs for the initial desktop workbench."""
from pathlib import Path
import math,copy,tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .cfu import default_options
from .config import read_preset
from .pipeline import scalar_options
from decimal import Decimal


def merge_to_display(value):return float((Decimal(str(value))-Decimal('.85'))/Decimal('.15'))
def merge_from_display(value):return float(Decimal('.85')+Decimal('.15')*Decimal(str(value)))


def center_dialog(dialog,parent):
    """Position a completed Tk dialog relative to its owning top-level."""
    parent=parent.winfo_toplevel();dialog.withdraw();dialog.transient(parent)
    dialog.update_idletasks()
    width=dialog.winfo_reqwidth();height=dialog.winfo_reqheight()
    x=parent.winfo_rootx()+(parent.winfo_width()-width)//2
    y=parent.winfo_rooty()+(parent.winfo_height()-height)//2
    # Keep the dialog on the desktop if it is larger than its parent.
    x=max(parent.winfo_vrootx(),x);y=max(parent.winfo_vrooty(),y)
    dialog.geometry(f'{width}x{height}+{x}+{y}')
    dialog.deiconify()


def entry(parent,label,variable,row,width=18):
    ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',padx=8,pady=5)
    widget=ttk.Entry(parent,textvariable=variable,width=width);widget.grid(row=row,column=1,sticky='ew',padx=8,pady=5)
    return widget


def file_entry(parent,label,variable,row,*,save=False,project=False):
    widget=entry(parent,label,variable,row,48)
    def browse():
        owner=parent.winfo_toplevel()
        if save:path=filedialog.asksaveasfilename(parent=owner,defaultextension='.aqua.h5',filetypes=[('AQuA2 Python Project','*.h5')])
        else:path=filedialog.askopenfilename(parent=owner,filetypes=[('Image / Parameters / MAT','*.tif *.tiff *.mat *.csv *.json *.avi'),('All Files','*.*')])
        if path:variable.set(path)
    ttk.Button(parent,text='Browse…',command=browse,width=8).grid(row=row,column=2,padx=5)
    return widget


class CFUForm:
    def __init__(self,parent,settings=None,*,dual_channel=False):
        settings=settings or default_options();settings={k:scalar_options(v) for k,v in settings.items()};self.values={};self.settings=copy.deepcopy(settings)
        definitions=[('cfuDetect','overlapThr1','Overlap Threshold',float),('cfuDetect','minNumEvt1','Min Events',int),
            ('cfuDetect','overlapThr2','Overlap Threshold CH2',float),('cfuDetect','minNumEvt2','Min Events CH2',int),
            ('cfuDetect','minimumCurveCorrelation','Merge Threshold',float),('cfuDetect','minimumCurveCorrelation2','Merge Threshold CH2',float)]
        self.definitions=definitions
        for row,(group,key,label,cast) in enumerate(definitions):
            value=settings.get(group,{}).get(key,.925 if key.startswith('minimumCurve') else 0)
            var=tk.StringVar(value=f'{merge_to_display(value):.6g}' if key.startswith('minimumCurve') else str(int(value)) if cast is int and float(value).is_integer() else str(value))
            self.values[key]=var;entry(parent,label,var,row)
        self.channel2_widgets=[widget for row,(_,key,*_) in enumerate(definitions) if key.endswith('2') for widget in parent.grid_slaves(row=row)]
        self.set_dual_channel(dual_channel)
        parent.columnconfigure(1,weight=1)
    def set_dual_channel(self,enabled):
        self.dual_channel=bool(enabled)
        for widget in self.channel2_widgets:
            if enabled:widget.grid()
            else:widget.grid_remove()
    def get(self):
        result=copy.deepcopy(self.settings)
        for group,key,label,cast in self.definitions:
            if key.endswith('2') and not self.dual_channel:continue
            value=float(self.values[key].get())
            if not math.isfinite(value) or (cast==int and value!=int(value)):raise ValueError(label+' must be a valid number')
            value=cast(value)
            if key.startswith('overlapThr') and not 0<=value<=1:raise ValueError(label+' must be 0–1')
            if key.startswith('minimumCurve'):
                if not 0<=value<=1:raise ValueError(label+' must be 0–1')
                original=self.settings.get(group,{}).get(key,.925)
                value=original if value==float(f'{merge_to_display(original):.6g}') else merge_from_display(value)
            if key in ('minNumEvt1','minNumEvt2','cfuNumThr') and value<1:raise ValueError(label+' must be at least 1')
            if key=='maxDist' and value<0:raise ValueError('Time window must be nonnegative.')
            if key=='pValueThr' and not 0<value<=1:raise ValueError('p-value must be (0,1]')
            result[group][key]=value
        return result


class NewAnalysisDialog(tk.Toplevel):
    def __init__(self,parent,callback):
        super().__init__(parent);self.title('Load Image · AQuA2');self.callback=callback;self.transient(parent)
        self.resizable(True,False);self.vars={key:tk.StringVar() for key in ('source','frames','regions','landmarks')}
        self.config_provider=getattr(parent,'input_settings',None)
        panel=ttk.Frame(self,padding=12);panel.pack(fill='both',expand=True)
        file_entry(panel,'Image / MAT',self.vars['source'],0)
        entry(panel,'Frame Range',self.vars['frames'],1)
        file_entry(panel,'Region mask MAT',self.vars['regions'],2)
        file_entry(panel,'Landmark MAT',self.vars['landmarks'],3)
        ttk.Button(panel,text='Cancel',command=self.destroy).grid(row=4,column=1,sticky='e',padx=8,pady=8)
        ttk.Button(panel,text='Import ',command=self.submit,style='Primary.TButton').grid(row=4,column=2,pady=8)
        center_dialog(self,parent);self.grab_set()
    def submit(self):
        try:
            source=self.vars['source'].get()
            if not Path(source).is_file():raise ValueError('Select a valid input file.')
            config=self.config_provider() if self.config_provider else {'preset':1,'gui_median':True}
            config={**config,'auto_channels':True}
            for key in ('regions','landmarks'):
                if self.vars[key].get():config[key]=self.vars[key].get()
            if self.vars['frames'].get():
                parts=self.vars['frames'].get().split(':')
                if not 2<=len(parts)<=3:raise ValueError('Frame range:  start:stop[:step]')
                config['frames']=[int(value) if value else None for value in parts]
                if len(parts)==3 and config['frames'][2] is not None and config['frames'][2]<=0:raise ValueError('Frame step must be positive.')
            self.callback({'operation':'load_input','source':source,'config':config});self.destroy()
        except Exception as error:messagebox.showerror('Import Failed',str(error),parent=self)


class ParameterDialog(tk.Toplevel):
    def __init__(self,parent,path,preset,callback):
        super().__init__(parent);self.title('Load Parameters');self.transient(parent);self.callback=callback
        self.path=tk.StringVar(value=str(path));self.preset=tk.StringVar(value=str(preset))
        panel=ttk.Frame(self,padding=12);panel.pack(fill='both',expand=True)
        file_entry(panel,'Parameter CSV',self.path,0);entry(panel,'Preset',self.preset,1)
        ttk.Button(panel,text='Cancel',command=self.destroy).grid(row=2,column=1,sticky='e',padx=8,pady=8)
        ttk.Button(panel,text='Load',command=self.submit,style='Primary.TButton').grid(row=2,column=2,pady=8);center_dialog(self,parent);self.grab_set()
    def submit(self):
        from .config import read_parameter_bundle
        try:
            path=Path(self.path.get());preset=int(self.preset.get());options,settings=read_parameter_bundle(path,preset)
            self.callback(path,preset,options,settings);self.destroy()
        except Exception as error:messagebox.showerror('Parameter Load Failed',str(error),parent=self)


class CFUSettingsDialog(tk.Toplevel):
    def __init__(self,parent,settings,callback,*,dual_channel=False):
        super().__init__(parent);self.title('CFU Detection');self.transient(parent);self.callback=callback
        panel=ttk.Frame(self,padding=12);panel.pack(fill='both',expand=True);self.form=CFUForm(panel,settings,dual_channel=dual_channel)
        ttk.Button(panel,text='Run  CFU',style='Primary.TButton',command=self.submit).grid(row=12,column=0,columnspan=2,pady=12);center_dialog(self,parent);self.grab_set()
    def submit(self):
        try:self.callback(self.form.get());self.destroy()
        except Exception as error:messagebox.showerror('Invalid Parameters',str(error),parent=self)
