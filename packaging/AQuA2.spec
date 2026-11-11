from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules,collect_data_files,collect_dynamic_libs

root=Path(SPECPATH).parent
source=root/'src'/'aqua2'
datas=collect_data_files('aqua2',include_py_files=True,excludes=['**/__pycache__/**','**/*.nbc','**/*.nbi','**/*.pyc'])
# Include source alongside bytecode so Numba can locate cacheable kernels.
datas += [(str(root/'LICENSE'),'.'),(str(root/'README.md'),'.'),(str(root/'THIRD_PARTY_NOTICES.md'),'.')]
datas += [(str(path),'licenses/'+str(path.relative_to(root/'licenses').parent)) for path in (root/'licenses').rglob('*') if path.is_file()]
binaries=[(str(p),'aqua2') for p in source.glob('*_core*.pyd')]
binaries+=collect_dynamic_libs('llvmlite')
hidden=[f'aqua2.{p.stem}' for p in source.glob('*.py') if p.stem!='__init__']
hidden+=collect_submodules('OpenGL.platform')+['OpenGL.GL','PyQt6.QtGui','PyQt6.QtCore']
# scikit-image uses lazy-loader stub files for its public submodule exports.
datas+=collect_data_files('skimage',includes=['**/*.pyi'])
a=Analysis([str(root/'packaging/entry.py')],pathex=[str(root/'src')],binaries=binaries,datas=datas,hiddenimports=hidden,
    excludes=['PySide2','PySide6','PyQt5','torch','tensorflow','IPython','jupyter','pytest','numba.tests','scipy.tests','matplotlib.tests'],
    hooksconfig={'matplotlib':{'backends':['TkAgg','Agg']}},runtime_hooks=[str(root/'packaging/runtime.py')],module_collection_mode={'aqua2':'py'},noarchive=False)
# The standard PyQt hook imports QtCore to register qt.conf in *every* process.
# Our pure-path hook preserves the tested architecture: Qt only in GPU worker.
a.scripts=[entry for entry in a.scripts if Path(entry[1]).name!='pyi_rth_pyqt6.py']
# Qt uses Windows' inbox ICU API (unsuffixed exports). Conda's ICU has renamed
# exports and shadows that OS component if copied into the application folder.
# Never redistribute/replace Windows ICU; leave resolution to System32.
a.binaries=[entry for entry in a.binaries if Path(entry[0]).name.lower() not in ('icuuc.dll','icuin.dll') and not Path(entry[0]).name.lower().startswith('icudt') and not entry[0].replace('\\','/').lower().startswith('opengl/dlls/')]
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='AQuA2',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,version=str(root/'packaging/version.txt'))
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='AQuA2-portable')
