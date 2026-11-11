"""Configure bundled Qt paths without importing Qt in the Tk/compute process."""
import os,sys
from pathlib import Path
plugins=Path(sys._MEIPASS)/'PyQt6/Qt6/plugins'
os.environ['QT_PLUGIN_PATH']=str(plugins)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(plugins/'platforms')
