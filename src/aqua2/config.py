"""Read the existing batch presets without changing parameter names or units."""

import csv
from pathlib import Path


def read_preset(path: str | Path, preset: int = 1, *, include_cfu=False) -> dict[str, float]:
    if isinstance(preset, bool) or not isinstance(preset, int) or preset < 1:
        raise ValueError("preset must be a positive integer")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        column = f"Preset{preset}"
        if column not in (reader.fieldnames or []):
            column = f"File{preset}"
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing Preset{preset}/File{preset} column")
        values = {}
        for row in reader:
            if not (row.get("Name") or "").strip():
                continue
            if row.get('Type','').strip().lower()=='cfu' and not include_cfu:continue
            key = (row.get("Variable") or "").strip()
            if not key or key in values:
                raise ValueError(f"Missing or duplicate parameter: {key!r}")
            values[key] = float(row[column])
    return values


def read_parameter_bundle(path, preset=1):
    """CSV stores algorithm values; GUI threshold mapping stays at the UI boundary."""
    from .cfu import default_options
    values=read_preset(path,preset,include_cfu=True);settings=default_options()
    defaults={'overlapThr':.5,'minNumEvt':3,'minimumCurveCorrelation':.925}
    for key,default in defaults.items():
        value=values.pop(key,default)
        if key=='minNumEvt':
            if not value>=1 or not float(value).is_integer():raise ValueError('Min events must be a positive integer.')
            value=int(value)
        elif not (.85 if key=='minimumCurveCorrelation' else 0)<=value<=1:raise ValueError('CFU threshold out of range')
        for suffix in ('1','2'):
            settings['cfuDetect'][key+suffix if key!='minimumCurveCorrelation' or suffix=='2' else key]=value
    return values,settings
