"""Call export_study(study, 'optuna.json') after your existing Optuna run."""
import json
from pathlib import Path

def export_study(study, filename='optuna.json'):
    names=getattr(study,'metric_names',None) or [f'Objective{i}' for i in range(len(study.directions))]
    trials=[]
    for t in study.trials:
        row={'number':t.number,'state':t.state.name,'params':t.params,'values':list(t.values) if t.values is not None else None}
        if t.datetime_start is not None and t.datetime_complete is not None:
            row.update(start=t.datetime_start.timestamp(),end=t.datetime_complete.timestamp())
        trials.append(row)
    data={'format':'deepcave-x-v1','study_name':study.study_name,'objectives':[{'name':name,'direction':d.name.lower()} for name,d in zip(names,study.directions)],'trials':trials}
    Path(filename).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    return filename
