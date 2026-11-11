"""Compact stage tabs, with shared CSV/Preset loading."""
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from .viewer_dialogs import CFUForm,merge_to_display
from .viewer_steps import (PREPROCESS_FIELDS,DETECTION_FIELDS,DETECTION_FLAGS,FIXED_DETECTION_FLAGS,
                           PRIMARY_PREPROCESS,PRIMARY_DETECTION,validated)
from .pipeline import scalar_options
from .config import read_parameter_bundle
from .cfu import default_options
from .workflow_steps import can_detect,phase

DEFAULT_PARAMETERS=Path(__file__).with_name('data')/'parameters_for_batch.csv'


class StepForms:
    def __init__(self,parent,run,show):
        self.tabs=ttk.Notebook(parent);self.tabs.pack(fill='x',pady=(4,8))
        self.values={};self.flags={};self.buttons={};self.entries=[];self.project=None;self.advanced={}
        for operation,title,fields,primary in [('preprocess','Preprocessing',PREPROCESS_FIELDS,PRIMARY_PREPROCESS),
                ('detect','Event Detection',DETECTION_FIELDS,PRIMARY_DETECTION),('cfu','CFU',[],())]:
            panel=ttk.Frame(self.tabs,padding=5);self.tabs.add(panel,text=title)
            if operation!='cfu':
                for key,*_ in fields:self.values[key]=tk.StringVar()
                basic=ttk.Frame(panel);basic.grid(row=0,column=0,columnspan=2,sticky='ew')
                advanced=ttk.Frame(panel);self.advanced[operation]=advanced
                advanced_keys=[key for key,*_ in fields if key not in primary and key not in ('northx','northy')] if operation=='preprocess' else ['sigThr','maxDelay','seedSzRatio']
                for group,keys in ((basic,primary),(advanced,advanced_keys)):
                    for row,key in enumerate(keys):
                        _,label,*_=next(field for field in fields if field[0]==key)
                        if key=='sigThr':label='Temporal Threshold'
                        var=self.values[key]
                        ttk.Label(group,text=label,wraplength=135).grid(row=row,column=0,sticky='w',pady=3)
                        widget=ttk.Entry(group,textvariable=var,width=8);widget.grid(row=row,column=1,sticky='ew',padx=(4,0),pady=3);self.entries.append(widget)
                    group.columnconfigure(1,weight=1)
                button=ttk.Button(panel,text='▶ Advanced');button.grid(row=1,column=0,columnspan=2,sticky='ew',pady=3)
                button.configure(command=lambda p=advanced,b=button:self.toggle_advanced(p,b))
                advanced.grid(row=2,column=0,columnspan=2,sticky='ew');advanced.grid_remove()
                row=3
                if operation=='detect':
                    for key,label,default in DETECTION_FLAGS:
                        var=tk.BooleanVar(value=default);self.flags[key]=var
                        widget=ttk.Checkbutton(panel,text=label,variable=var);widget.grid(row=row,column=0,columnspan=2,sticky='w');self.entries.append(widget);row+=1
            else:
                self.cfu=CFUForm(panel);row=8
                for child in panel.winfo_children():
                    if isinstance(child,ttk.Entry):child.configure(width=8);self.entries.append(child)
                    elif isinstance(child,ttk.Label):child.configure(wraplength=125)
            panel.columnconfigure(1,weight=1)
            self.buttons[operation]=ttk.Button(panel,text='Run '+title,command=lambda op=operation:run(op),style='Primary.TButton')
            self.buttons[operation].grid(row=row,column=0,columnspan=2,sticky='ew',pady=(5,3))
            ttk.Button(panel,text='Show '+title,command=lambda op=operation:show(op)).grid(row=row+1,column=0,columnspan=2,sticky='ew')
        self.defaults,self.default_cfu=read_parameter_bundle(DEFAULT_PARAMETERS)
        self.apply_parameters(self.defaults,self.default_cfu);self.update()

    @staticmethod
    def toggle_advanced(panel,button):
        expanded=bool(panel.grid_info())
        if expanded:panel.grid_remove()
        else:panel.grid()
        button.configure(text=('▶' if expanded else '▼')+' Advanced')

    def apply_parameters(self,options,settings):
        for key,var in self.values.items():var.set(str(options.get(key,self.defaults[key])))
        for key,label,default in DETECTION_FLAGS:self.flags[key].set(bool(options.get(key,default)))
        self.set_cfu(settings)

    def load(self,project,operation=None):
        self.project=project;opts=project.opts;self.cfu.set_dual_channel('channel2' in project.channels)
        partial=operation in ('preprocess','detect','cfu')
        keys={key for key,*_ in (PREPROCESS_FIELDS if operation=='preprocess' else DETECTION_FIELDS)} if partial else set(self.values)
        if operation=='cfu':keys=set()
        for key,var in self.values.items():
            if key in keys:var.set(str(opts.get(key,self.defaults[key])))
        if not partial or operation=='detect':
            for key,label,default in DETECTION_FLAGS:self.flags[key].set(bool(opts.get(key,default)))
        settings=project.data['stages'].get('cfu',{}).get('cfuOpts') or project.data.get('config',{}).get('cfu_settings') or default_options()
        if not partial or operation=='cfu':self.set_cfu(settings)
        self.tabs.select({'input':0,'preprocess':1,'detection':2,'cfu':2}[phase(project)]);self.update()

    def set_cfu(self,settings):
        self.cfu.settings={key:scalar_options(value) for key,value in settings.items()}
        for group,key,label,cast in self.cfu.definitions:
            value=self.cfu.settings.get(group,{}).get(key,.925 if key.startswith('minimumCurve') else self.cfu.values[key].get())
            self.cfu.values[key].set(f'{merge_to_display(value):.6g}' if key.startswith('minimumCurve') else str(int(float(value))) if cast is int and float(value).is_integer() else str(value))

    def get(self,operation):
        if operation=='cfu':return self.cfu.get()
        fields=PREPROCESS_FIELDS if operation=='preprocess' else DETECTION_FIELDS
        result=validated({key:self.values[key].get() for key,*_ in fields},fields)
        if operation=='detect':result.update(FIXED_DETECTION_FLAGS);result.update({key:var.get() for key,var in self.flags.items()})
        return result

    def update(self,busy=False):
        p=self.project;allowed=dict(preprocess=p is not None,detect=can_detect(p),cfu=bool(p and 'detection' in p.data['stages']))
        for key,button in self.buttons.items():button.configure(state='normal' if allowed[key] and not busy else 'disabled')
        for entry in self.entries:entry.configure(state='disabled' if busy else 'normal')
