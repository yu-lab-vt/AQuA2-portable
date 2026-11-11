"""Locate compatible Tcl scripts in Conda/venv installations when needed.

Only this process's environment is adjusted; no system settings or installs.
Normal Python installations take the first path without any adjustment.
"""
from pathlib import Path
import os,sys
import tkinter as tk


def synchronize_canvas(canvas):
    """Size embedded TkAgg's backing image after its initial DPI update."""
    if sys.platform != 'win32':return
    widget = canvas.get_tk_widget()
    # TkAgg 3.10 changes figure DPI on <Map>. A pack-managed canvas may keep
    # its allocated size without another <Configure>, leaving an oversized
    # backing image. Reuse its normal resize handler with actual Tk pixels.
    widget.bind('<Map>', lambda event: widget.after_idle(lambda: widget.event_generate(
        '<Configure>', width=widget.winfo_width(), height=widget.winfo_height())), add='+')


def initialize(root):
    # Embedded TkAgg bypasses Matplotlib's window manager, which normally
    # enables Windows DPI awareness before creating Tk. Use the same helper
    # here so Windows does not bitmap-stretch the entire workbench at 150%.
    if sys.platform == 'win32':
        from matplotlib import _c_internal_utils
        _c_internal_utils.Win32_SetProcessDpiAwareness_max()
    try:tk.Tk.__init__(root);return
    except tk.TclError as original:
        if 'init.tcl' not in str(original) and 'tk.tcl' not in str(original):raise
        base=Path(sys.base_prefix)
        candidates=[base/'tcl'/'tcl8.6',base/'Library'/'lib'/'tcl8.6']
        candidates.extend(sorted((base/'pkgs').glob('tk-*/Library/lib/tcl8.6')))
        before={key:os.environ.get(key) for key in ('TCL_LIBRARY','TK_LIBRARY')}
        for directory in candidates:
            if not (directory/'init.tcl').is_file():continue
            os.environ['TCL_LIBRARY']=str(directory);os.environ['TK_LIBRARY']=str(directory.parent/'tk8.6')
            try:tk.Tk.__init__(root);return
            except tk.TclError:continue
        for key,value in before.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        raise tk.TclError(str(original)+'\nInstall matching Tcl/Tk scripts for this Python interpreter.') from original
