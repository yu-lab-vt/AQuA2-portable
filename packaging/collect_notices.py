"""Retain installed dependency notices verbatim; no relicensing is asserted."""
from importlib import metadata
from pathlib import Path
import json,re,shutil,sys

root=Path(__file__).resolve().parents[1];out=root/'licenses';out.mkdir(exist_ok=True)
names=[line.split('==')[0] for line in (root/'requirements-runtime.lock').read_text().splitlines() if '==' in line]+['pyinstaller']
records=[]
for name in names:
    dist=metadata.distribution(name);files=[]
    for item in dist.files or []:
        if not re.search(r'(licen[sc]e|copying|notice)',Path(str(item)).name,re.I):continue
        source=Path(dist.locate_file(item))
        if not source.is_file():continue
        relative=Path(str(item).replace('../',''))
        target=out/dist.metadata['Name']/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        files.append(target.relative_to(root).as_posix())
    records.append(dict(name=dist.metadata['Name'],version=dist.version,declared_license=dist.metadata.get('License-Expression') or dist.metadata.get('License','Not declared'),notices=files,homepage=dist.metadata.get('Home-page','')))
for path,name in ((Path(sys.base_prefix)/'LICENSE.txt','CPython-LICENSE.txt'),(Path(sys.base_prefix)/'Library/lib/tcl8.6/license.terms','Tcl-license.terms'),(Path(sys.base_prefix)/'Library/lib/tk8.6/license.terms','Tk-license.terms')):
    if path.is_file():shutil.copy2(path,out/name)
for folder,name in (('tcl8.6','Tcl-license.terms'),('tk8.6','Tk-license.terms')):
    path=Path(sys.base_prefix)/'tcl'/folder/'license.terms'
    if path.is_file():shutil.copy2(path,out/name)
python_license=Path(sys.base_prefix)/'LICENSE_PYTHON.txt'
if python_license.is_file():shutil.copy2(python_license,out/'CPython-LICENSE.txt')
if not (out/'Tcl-license.terms').exists():
    candidates=sorted((Path(sys.base_prefix)/'pkgs').glob('tk-8.6.13-*/info/licenses/tcl8.6.13/license.terms'))
    if candidates:shutil.copy2(candidates[0],out/'Tcl-license.terms')
(out/'dependency-inventory.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print(f'Retained notices for {len(records)} dependencies')
