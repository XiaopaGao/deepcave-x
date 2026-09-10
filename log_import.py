"""Bounded data-only imports into DeepCAVE's Run API; never unpickle uploads."""
import base64
import csv
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from ConfigSpace import ConfigurationSpace, Categorical, Float, Constant
from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.converters.dataframe import DataFrameRun
from deepcave.runs.converters.smac3v2 import SMAC3v2Run

MAX_BYTES = 8 * 1024 * 1024
MAX_TRIALS = 5000
MAX_PARAMS = 64

def number(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('评分或时间含有 NaN / Infinity，请导出有限数值。')
    return x

def read_json(data):
    return json.loads(data.decode('utf-8-sig'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('JSON 不能包含 NaN / Infinity。')))

def clean_params(params):
    if not isinstance(params, dict) or not params or len(params) > MAX_PARAMS:
        raise ValueError('每条成功试验应含 1–64 个超参数。')
    result = {}
    for key, value in params.items():
        if not isinstance(key,str) or len(key)>100:
            raise ValueError('超参数名称不合法。')
        if isinstance(value,(dict,list)) or (value is not None and not isinstance(value,(str,int,float,bool))):
            raise ValueError('超参数只支持数字、布尔值、字符串或空值。')
        if isinstance(value,str) and len(value)>300:
            raise ValueError('类别值过长。')
        if isinstance(value,float): number(value)
        if value is not None: result[key]=value
    return result

def make_run(name, records, objectives, meta):
    if not records or len(records)>MAX_TRIALS:
        raise ValueError('日志应有 1–5000 条成功试验。')
    names=sorted(set(k for row in records for k in row['params']))
    if not names or len(names)>MAX_PARAMS: raise ValueError('超参数总数应为 1–64。')
    space=ConfigurationSpace()
    dimensions=0
    for key in names:
        values=[row['params'][key] for row in records if key in row['params']]
        numeric=all(isinstance(v,(float,int)) and not isinstance(v,bool) for v in values)
        if numeric:
            lo,hi=min(values),max(values)
            hp=Constant(key,lo) if lo==hi else Float(key,(lo,hi))
            dimensions+=2
        else:
            values=list(dict.fromkeys(str(v) for v in values))
            if len(values)>128: raise ValueError('单个类别超参数最多支持 128 个已观测取值。')
            hp=Constant(key,values[0]) if len(values)==1 else Categorical(key,values)
            dimensions+=len(values)+1
        space.add(hp)
    if dimensions>512: raise ValueError('编码后维度超过 512，请减少类别数量或超参数。')
    run=DataFrameRun(name=name,configspace=space,objectives=objectives,meta=meta)
    for i,row in enumerate(records):
        run.add(costs=row['values'],config=row['params'],seed=i,
                start_time=row.get('start',i),end_time=row.get('end',i),
                budget=row.get('budget',1),additional={'source_trial':row.get('number',i)})
    return run

def csv_run(raw,name,direction):
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    fields=reader.fieldnames or []
    if len(fields)>150: raise ValueError('CSV 列过多。')
    params=[f for f in fields if f.startswith('params_')]
    objectives=['value'] if 'value' in fields else sorted([f for f in fields if f.startswith('values_')], key=lambda f:int(f[7:]))
    if not params or not objectives: raise ValueError('Optuna CSV 需要 value（或 values_0 等）与 params_ 开头的参数列；可用 study.trials_dataframe().to_csv(..., index=False) 导出。')
    records=[]; skipped=0
    for i,row in enumerate(reader):
        if i>=MAX_TRIALS: raise ValueError('单份日志最多支持 5000 条试验。')
        if row.get('state','COMPLETE').upper() not in ('COMPLETE','SUCCESS'):
            skipped+=1; continue
        p={}
        for f in params:
            value=row[f]
            if value in ('',None): continue
            try: value=number(value)
            except ValueError: pass
            p[f[7:]]=value
        records.append({'params':clean_params(p),'values':[number(row[f]) for f in objectives], 'number':row.get('number',i),'start':i,'end':i})
    meta={'framework':'Optuna CSV','skipped':skipped,'time_available':False,'ordering':'CSV 行顺序；请按试验顺序导出','space_inferred':True}
    return make_run(name,records,[Objective(f,optimize=direction) for f in objectives],meta)

def json_run(raw,name,direction):
    data=read_json(raw)
    if not isinstance(data,dict) or data.get('format')!='deepcave-x-v1': raise ValueError('JSON 应使用本站提供的 Optuna 导出函数生成（format=deepcave-x-v1）。SMAC 请上传日志 ZIP。')
    objectives=data.get('objectives',[])
    if not 1<=len(objectives)<=10: raise ValueError('需要 1–10 个目标指标。')
    objs=[Objective(str(o['name'])[:100],optimize={'maximize':'upper','minimize':'lower','upper':'upper','lower':'lower'}[o['direction']]) for o in objectives]
    trials=data.get('trials',[])
    if not isinstance(trials,list) or len(trials)>MAX_TRIALS: raise ValueError('单份日志最多支持 5000 条试验。')
    records=[]; skipped=0; times=True
    for i,t in enumerate(trials):
        if t.get('state','COMPLETE') not in ('COMPLETE','SUCCESS'):
            skipped+=1; continue
        values=t.get('values',[t.get('value')])
        if len(values)!=len(objs): raise ValueError('试验评分数与目标指标数不一致。')
        timed='start' in t and 'end' in t
        times=times and timed
        start,end=number(t.get('start',i)),number(t.get('end',i))
        if end<start: raise ValueError('结束时间不能早于开始时间。')
        records.append({'params':clean_params(t['params']),'values':[number(v) for v in values], 'number':t.get('number',i),'start':start,'end':end,'budget':number(t.get('budget',1))})
    if not times:
        for i,row in enumerate(records): row['start']=row['end']=i
    meta={'framework':'Optuna JSON','skipped':skipped,'time_available':times,'space_inferred':True}
    return make_run(name,records,objs,meta)

def smac_run(raw,name):
    required={'configspace.json','scenario.json','runhistory.json'}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist())>100: raise ValueError('ZIP 文件项过多，只需压缩一个运行目录中的三个 JSON 文件。')
        chosen={}; total=0
        for info in archive.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename: raise ValueError('ZIP 路径不合法。')
            total+=info.file_size
            if total>32*1024*1024: raise ValueError('ZIP 解压后的总大小不能超过 32 MB。')
            if p.name in required:
                if p.name in chosen: raise ValueError('ZIP 包含多个运行，请将各运行分别压缩上传。')
                chosen[p.name]=info
        if set(chosen)!=required: raise ValueError('SMAC ZIP 必须含 configspace.json、scenario.json、runhistory.json。')
        # SMAC writes Infinity for unlimited runtime / crash-cost metadata.
        # Allow that native convention; successful scores are checked below.
        docs={k:json.loads(archive.read(v).decode('utf-8-sig')) for k,v in chosen.items()}
    rh=docs['runhistory.json']; data=rh.get('data',[])
    if not isinstance(data,list) or not 1<=len(data)<=MAX_TRIALS: raise ValueError('支持 SMAC 2.x 日志，试验条数应为 1–5000。')
    if len(docs['configspace.json'].get('hyperparameters',[]))>MAX_PARAMS: raise ValueError('最多支持 64 个超参数。')
    for params in rh.get('configs',{}).values(): clean_params(params)
    # Write exactly three data files with fixed names, never archive paths.
    with tempfile.TemporaryDirectory(prefix='deepcave-x-') as folder:
        for filename,doc in docs.items(): Path(folder,filename).write_text(json.dumps(doc),encoding='utf-8')
        run=SMAC3v2Run.from_path(Path(folder))
    run.name=name
    run.meta.update({'framework':'SMAC 2.x','time_available':True,'space_inferred':False})
    successful=0
    for trial in run.get_trials():
        if np.isnan(float(trial.budget)):
            raise ValueError('预算不能为 NaN。')
        if trial.status==Status.SUCCESS:
            successful+=1
            for cost in trial.costs:
                if cost is not None: number(cost)
    if not successful: raise ValueError('日志中没有成功试验。')
    run.meta['skipped']=len(list(run.get_trials()))-successful
    return run

def import_file(content,filename,direction='upper',label=''):
    if direction not in ('lower','upper'): raise ValueError('请选择优化方向。')
    if not isinstance(content,str) or len(content)>MAX_BYTES*4//3+300: raise ValueError('单文件不能超过 8 MB。')
    raw=base64.b64decode(content.split(',',1)[1],validate=True)
    if len(raw)>MAX_BYTES: raise ValueError('单文件不能超过 8 MB。')
    name=(label.strip() or Path(filename).stem)[:80]
    suffix=Path(filename).suffix.lower()
    if suffix=='.csv': run=csv_run(raw,name,direction)
    elif suffix=='.json': run=json_run(raw,name,direction)
    elif suffix=='.zip': run=smac_run(raw,name)
    else: raise ValueError('支持 .csv、.json 和 .zip；不接受 .pkl、模型文件或原始数据集。')
    return run
