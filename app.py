"""DeepCAVE-X upload workspace: data-only imports, per-browser sessions."""
import os
import secrets
import time
import uuid
from threading import RLock
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import session
from dash import Dash, dcc, html, Input, Output, State, no_update
from deepcave.runs import Status
from log_import import import_file
from dashboard_config.generic import importance
from dashboard_config.stagnation import find_stagnation_segments, make_diagnosis

ROOT=Path(__file__).resolve().parent
app=Dash(__name__,title='DeepCAVE-X · 优化日志分析',suppress_callback_exceptions=True)
server=app.server
server.secret_key=os.environ.get('DEEPCAVE_SECRET') or secrets.token_hex(32)
server.config.update(MAX_CONTENT_LENGTH=24*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax')
LOCK=RLock(); SESSIONS={}; TTL=7200

@server.before_request
def prepare_session():
    now=time.monotonic()
    with LOCK:
        for sid in list(SESSIONS):
            if now-SESSIONS[sid]['seen']>TTL: del SESSIONS[sid]
        sid=session.get('workspace')
        if sid not in SESSIONS:
            if len(SESSIONS)>=30: return '试用服务当前已满，请稍后重试。',503
            sid=secrets.token_urlsafe(32);session['workspace']=sid
            SESSIONS[sid]={'seen':now,'runs':{}}
        SESSIONS[sid]['seen']=now

@server.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Robots-Tag']='noindex, nofollow'
    response.headers['Cache-Control']='no-store'
    return response

def registry():
    with LOCK: return dict(SESSIONS[session['workspace']]['runs'])

def msg(text): return html.P(text,className='notice')

def controls(side):
    return html.Section([
        html.H3('运行 '+side.upper()),
        html.Label('已上传日志',htmlFor='run-'+side),
        dcc.Dropdown(id='run-'+side,options=[],placeholder='请先上传日志',clearable=False),
        html.Label('评价指标',htmlFor='objective-'+side),
        dcc.Dropdown(id='objective-'+side,options=[],clearable=False),
        html.Label('资源预算（分别分析，避免混合不同训练预算）',htmlFor='budget-'+side),
        dcc.Dropdown(id='budget-'+side,options=[],clearable=False),
        html.Label('指标显示方式',htmlFor='transform-'+side),
        dcc.Dropdown(id='transform-'+side,options=[{'label':'原始指标（按日志方向寻找最优）','value':'raw'},{'label':'错误率转准确率：1 − 错误率','value':'accuracy'}],value='raw',clearable=False),
    ],className='control-card')

app.layout=html.Div([
    dcc.Location(id='location'),
    html.Header([html.Strong('DeepCAVE-X'),html.Span('上传优化日志 · 比较搜索过程 · 诊断瓶颈')]),
    html.Main([
        html.H1('分析你自己的优化实验'),
        html.P('在自己的数据集上运行 Optuna 或 SMAC，再上传优化日志。每位访问者拥有独立分析空间。'),
        html.Details([html.Summary('上传什么文件？查看导出指引'),dcc.Markdown('''**Optuna：** 上传 `study.trials_dataframe().to_csv("optuna.csv", index=False)` 导出的 CSV，并在下方选择指标方向。多目标 CSV 的所有指标将使用同一方向；方向不同时请用 JSON 导出。\n\n**Optuna JSON：** 下载下方导出函数，在现有优化代码末尾调用 `export_study(study, "optuna.json")`，保留每个目标的方向和试验时间。\n\n**SMAC 2.x：** 将一次运行的 `configspace.json`、`scenario.json`、`runhistory.json` 打包成一个 ZIP。一次运行一个 ZIP。SMAC 的 cost 默认越低越好；只有确认为错误率时，才使用“1 − 错误率”转换。\n\n上传的是超参数和评分日志，不是原始训练数据、模型文件或 `.pkl`。单文件最多 8 MB、5000 条试验、64 个超参数。上传后保存在服务内存；连续两小时无访问或服务重启后清除。'''),
            html.A('下载 Optuna JSON 导出函数',href='/assets/export_optuna.py',download='export_optuna.py')]),
        html.Div([
            html.Div([html.Label('数据集／任务标识（用于避免跨任务误比较）',htmlFor='dataset'),dcc.Input(id='dataset',type='text',placeholder='例如 my_dataset_v1 / split42',maxLength=100)]),
            html.Div([html.Label('CSV 中的指标方向',htmlFor='csv-direction'),dcc.Dropdown(id='csv-direction',options=[{'label':'越高越好（准确率、AUC 等）','value':'upper'},{'label':'越低越好（loss、RMSE 等）','value':'lower'}],value='upper',clearable=False)])
        ],className='two-cols'),
        dcc.Upload(id='upload',children=html.Div(['拖放优化日志到这里，或 ',html.Strong('选择文件'),'（CSV / JSON / ZIP，可多选）']),multiple=True,accept='.csv,.json,.zip',max_size=8*1024*1024,className='upload-zone'),
        dcc.Loading(html.Div(id='upload-status')),
        html.Div(id='inventory'),
        html.Div([controls('a'),controls('b')],className='two-cols'),
        html.P('重要性分析和瓶颈诊断使用运行 A；轨迹对比同时使用 A 和 B。只上传一份日志也能分析。',className='muted'),
        dcc.Tabs(id='analysis-tab',value='trajectory',children=[
            dcc.Tab(label='轨迹对比',value='trajectory',children=[
                dcc.Checklist(id='confirmed',options=[{'label':'我已核对两次运行的数据集、划分、评价指标及搜索预算可比较','value':'yes'}],value=[]),
                dcc.Checklist(id='show-trials',options=[{'label':'显示单次试验成绩和超参数','value':'yes'}],value=['yes']),
                dcc.RadioItems(id='layout-mode',options=[{'label':'叠加比较','value':'overlay'},{'label':'并排比较','value':'side'}],value='overlay',labelStyle={'display':'inline-block','marginRight':'20px'}),
                dcc.Loading(dcc.Graph(id='trajectory-chart')),html.Div(id='trajectory-note')]),
            dcc.Tab(label='超参数重要性',value='importance',children=[dcc.Loading(dcc.Graph(id='importance-chart')),html.Div(id='importance-note'),html.P('使用随机森林回归的 MDI 重要性。至少需要 10 条、评分有变化的成功试验；结果受采样范围和参数类别数量影响，不是因果贡献或 Sobol 指数。')]),
            dcc.Tab(label='瓶颈诊断',value='stagnation',children=[
                html.Label('耐心值：连续多少次没有明显改善才标记停滞'),dcc.Slider(id='patience',min=2,max=30,step=1,value=5,marks={i:str(i) for i in [2,5,10,15,20,25,30]}),
                html.Label('最小改善值（使用当前指标的数值单位）',htmlFor='min-delta'),dcc.Input(id='min-delta',type='number',value=0.0001,min=0,step=0.0001),
                dcc.Loading(dcc.Graph(id='stagnation-chart')),html.Div(id='stagnation-note'),html.Div(id='stagnation-table'),
                html.P('这是基于已上传日志的阈值检测。调整参数会重新分析；不会实时接收训练进程。停滞不能单独证明陷入局部最优，建议也未经后续实验验证。')]),
        ]),
    ]),html.Footer('DeepCAVE-X · 基于 DeepCAVE Run API · 用户反馈请在线下交给作业作者')])

@app.callback(Output('analysis-tab','value'),Input('location','pathname'))
def route(path):
    return 'importance' if (path or '').endswith('importance') else ('stagnation' if (path or '').endswith('stagnation') else 'trajectory')

@app.callback([Output('upload-status','children'),Output('inventory','children'),Output('run-a','options'),Output('run-a','value'),Output('run-b','options'),Output('run-b','value')],
    Input('upload','contents'),State('upload','filename'),State('csv-direction','value'),State('dataset','value'),State('run-a','value'),State('run-b','value'))
def upload(contents,filenames,direction,dataset,a,b):
    reports=[]
    current=registry()
    if contents:
        if not dataset or not dataset.strip():
            reports.append(msg('请先填写数据集／任务标识，再重新选择文件。'))
        elif len(contents)>10 or len(current)+len(contents)>10:
            reports.append(msg('一个分析空间最多上传 10 份日志。'))
        else:
            for content,filename in zip(contents,filenames):
                try:
                    run=import_file(content,filename,direction)
                    run.meta['task_label']=dataset.strip()[:100]
                    key=uuid.uuid4().hex
                    with LOCK: SESSIONS[session['workspace']]['runs'][key]=run
                    reports.append(msg(filename+'：导入成功。'))
                except Exception as exc:
                    # Never return local paths or tracebacks to a visitor.
                    detail=str(exc) if isinstance(exc,ValueError) else '日志格式无法读取，请对照导出指引检查文件。'
                    reports.append(msg(filename+'：'+detail[:300]))
    current=registry()
    options=[{'label':r.name+' · '+r.meta.get('task_label',''),'value':k} for k,r in current.items()]
    keys=list(current)
    a=a if a in current else (keys[0] if keys else None)
    b=b if b in current else (keys[1] if len(keys)>1 else None)
    table=html.Table([html.Thead(html.Tr([html.Th(t) for t in ['日志','任务','来源','试验数','跳过的非成功试验']])),html.Tbody([
        html.Tr([html.Td(r.name),html.Td(r.meta.get('task_label','')),html.Td(r.meta.get('framework','')),html.Td(len(list(r.get_trials()))),html.Td(r.meta.get('skipped',0))]) for r in current.values()])]) if current else msg('尚未上传日志。先上传一份进行分析，或上传两份比较。')
    return reports,table,options,a,options,b

for side in ['a','b']:
    def register(side):
        @app.callback([Output('objective-'+side,'options'),Output('objective-'+side,'value'),Output('budget-'+side,'options'),Output('budget-'+side,'value')],Input('run-'+side,'value'))
        def choices(key):
            run=registry().get(key)
            if run is None:return [],None,[],None
            options=[{'label':o.name+('（越低越好）' if o.optimize=='lower' else '（越高越好）'),'value':i} for i,o in enumerate(run.get_objectives()) if o.name!='Time']
            budgets=sorted({float(t.budget) for t in run.get_trials()})
            bs=[{'label':'无分层预算' if not np.isfinite(v) else str(v),'value':str(v)} for v in budgets]
            return options,options[0]['value'] if options else None,bs,bs[-1]['value'] if bs else None
    register(side)

def rows_for(key,obj,budget,transform):
    run=registry().get(key)
    if run is None or obj is None:raise ValueError('请上传并选择运行与评价指标。')
    obj=int(obj);objective=run.get_objectives()[obj]
    lower=objective.optimize=='lower'
    configs=run.get_configs(); rows=[]
    for t in sorted(run.get_trials(),key=lambda t:(t.end_time,t.start_time)):
        if budget is not None and float(t.budget)!=float(budget):continue
        if t.status!=Status.SUCCESS or t.costs[obj] is None:continue
        v=float(t.costs[obj])
        if not np.isfinite(v):continue
        if transform=='accuracy':
            if not lower or not 0<=v<=1:raise ValueError('错误率转准确率需要越低越好的 0–1 错误率日志。')
            v=1-v
        rows.append((len(rows)+1,v,dict(configs[t.config_id])))
    if not rows:raise ValueError('当前指标与预算下没有成功试验。')
    if transform=='accuracy':lower=False
    label='准确率（1 − 错误率）' if transform=='accuracy' else objective.name
    return run,rows,lower,label

def curve(fig,rows,lower,name,color,show=True,col=None):
    x=[r[0] for r in rows];y=np.array([r[1] for r in rows]);best=np.minimum.accumulate(y) if lower else np.maximum.accumulate(y)
    kwargs={'row':1,'col':col} if col else {}
    fig.add_trace(go.Scatter(x=x,y=best,mode='lines+markers',name=name+' · 历史最优',line={'color':color,'width':3,'shape':'hv'}),**kwargs)
    if show:fig.add_trace(go.Scatter(x=x,y=y,mode='markers',name=name+' · 单次试验',marker={'color':color,'opacity':.5},text=[str(r[2]) for r in rows],hovertemplate='成功试验序号：%{x}<br>评分：%{y:.6g}<br>%{text}<extra></extra>'),**kwargs)

def layout(fig,label='评分'):
    fig.update_layout(template='plotly_white',height=480,margin={'t':45,'l':65,'r':25,'b':60},legend={'orientation':'h'})
    fig.update_xaxes(title_text='当前预算下的成功试验序号（按完成顺序）')
    fig.update_yaxes(title_text=label)
    return fig

@app.callback([Output('trajectory-chart','figure'),Output('trajectory-note','children')],
    [Input('run-a','value'),Input('objective-a','value'),Input('budget-a','value'),Input('transform-a','value'),Input('run-b','value'),Input('objective-b','value'),Input('budget-b','value'),Input('transform-b','value'),Input('confirmed','value'),Input('show-trials','value'),Input('layout-mode','value')])
def trajectory(a,oa,ba,ta,b,ob,bb,tb,confirmed,show,mode):
    try:
        if 'yes' not in (confirmed or []):raise ValueError('请选择 A / B，并核对、勾选实验条件可比较。')
        if a==b:raise ValueError('请选择两份不同的日志。')
        ra,xa,la,na=rows_for(a,oa,ba,ta);rb,xb,lb,nb=rows_for(b,ob,bb,tb)
        if ra.meta.get('task_label')!=rb.meta.get('task_label'):raise ValueError('两份日志的任务标识不同，不能直接比较。')
        if la!=lb:raise ValueError('两侧指标方向不同。若 A 是准确率、B 是错误率，请对 B 使用错误率转准确率。否则请重新选择可比较指标。')
        fig=make_subplots(rows=1,cols=2,shared_yaxes=True,subplot_titles=[ra.name,rb.name]) if mode=='side' else go.Figure()
        for i,(r,xx,ll,c) in enumerate([(ra,xa,la,'#2878b5'),(rb,xb,lb,'#e87924')],1):curve(fig,xx,ll,r.name,c,'yes' in (show or []),i if mode=='side' else None)
        label=na if na==nb else na+' / '+nb
        best=lambda xx,ll:min(r[1] for r in xx) if ll else max(r[1] for r in xx)
        note=f'A：{len(xa)} 条成功试验，最优 {best(xa,la):.6g}；B：{len(xb)} 条成功试验，最优 {best(xb,lb):.6g}。'+('越低越好。' if la else '越高越好。')+'试验次数不代表运行耗时；结论仅适用于当前实验。'
        return layout(fig,label),msg(note)
    except (ValueError,KeyError,IndexError,TypeError) as e:return layout(go.Figure()),msg(str(e))

@app.callback([Output('importance-chart','figure'),Output('importance-note','children')],Input('run-a','value'),Input('objective-a','value'),Input('budget-a','value'),Input('transform-a','value'))
def important(a,o,b,t):
    try:
        run,rows,lower,label=rows_for(a,o,b,t)
        # Per-run cache stays inside this visitor's private workspace.
        key=(o,b,t);cache=run.__dict__.setdefault('_x_importance_cache',{})
        if key not in cache: cache[key]=importance(rows)
        scores=cache[key]
        fig=go.Figure(go.Bar(x=list(scores.values())[::-1],y=list(scores)[::-1],orientation='h',marker_color='#2878b5'))
        fig.update_layout(template='plotly_white',height=max(360,28*len(scores)),xaxis_title='归一化 MDI 重要性',margin={'l':180,'r':30,'t':30,'b':50})
        return fig,msg(f'{run.name}：基于 {len(rows)} 条成功试验。最重要的参数为 {next(iter(scores))}，估计重要性 {next(iter(scores.values())):.4f}。')
    except (ValueError,KeyError,IndexError,TypeError) as e:return layout(go.Figure()),msg(str(e))

@app.callback([Output('stagnation-chart','figure'),Output('stagnation-note','children'),Output('stagnation-table','children')],Input('run-a','value'),Input('objective-a','value'),Input('budget-a','value'),Input('transform-a','value'),Input('patience','value'),Input('min-delta','value'))
def stagnation(a,o,b,t,patience,delta):
    try:
        run,rows,lower,label=rows_for(a,o,b,t)
        delta=float(delta);patience=int(patience)
        if not np.isfinite(delta) or delta<0 or not 2<=patience<=30:raise ValueError('请使用 2–30 的耐心值及非负、有限的最小改善值。')
        ys=[-r[1] if lower else r[1] for r in rows]
        segments=find_stagnation_segments(ys,patience,delta)
        fig=go.Figure();curve(fig,rows,lower,run.name,'#2878b5')
        for start,end in segments:fig.add_vrect(x0=start-.5,x1=end+.5,fillcolor='#e15759',opacity=.16,line_width=0)
        table=html.Table([html.Thead(html.Tr([html.Th(v) for v in ['停滞开始序号','结束序号','持续次数']])),html.Tbody([html.Tr([html.Td(s),html.Td(e),html.Td(e-s+1)]) for s,e in segments])]) if segments else msg('没有达到当前阈值的停滞区间；不代表后续一定持续改善。')
        return layout(fig,label),msg(make_diagnosis(segments,len(rows))),table
    except (ValueError,KeyError,IndexError,TypeError) as e:return layout(go.Figure()),msg(str(e)),''

if __name__=='__main__': app.run_server(host='127.0.0.1',port=8051,debug=False)
