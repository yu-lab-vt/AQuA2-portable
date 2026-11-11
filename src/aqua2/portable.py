"""Frozen entry point support. No Qt imports in the Tk/compute parent."""
import os,sys,json,traceback
from pathlib import Path


def configure_runtime():
    if not getattr(sys,'frozen',False):return
    root=Path(os.environ.get('LOCALAPPDATA',Path.home()))/'AQuA2-portable'
    cache=root/'cache';logs=root/'logs';cache.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(cache))
    # Windowed executables have no console streams, including spawned workers.
    if sys.stdout is None:sys.stdout=(logs/f'process-{os.getpid()}.log').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=sys.stdout


def main():
    configure_runtime()
    if len(sys.argv)>2 and sys.argv[1]=='--verify':
        from .portable_check import verify
        return verify(Path(sys.argv[2]))
    if len(sys.argv)>1 and sys.argv[1]=='--cli':
        sys.argv.pop(1)
        from .__main__ import main as cli
        return cli()
    try:
        from .viewer import launch
        launch(sys.argv[1] if len(sys.argv)>1 else None)
        return 0
    except Exception:
        traceback.print_exc()
        if getattr(sys,'frozen',False):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,'AQuA2 could not start. See %LOCALAPPDATA%/AQuA2-portable/logs for details.','AQuA2',16)
        return 1
