import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from dash import dcc, html
from dash.dependencies import Input, Output
from deepcave.plugins import Plugin
from deepcave.runs import Status

def runs():
    from deepcave import run_handler
    return {str(r.path): r for r in run_handler.get_runs()}

def extract(run, objective, scale):
    objective = int(objective)
    if run.get_objectives()[objective].optimize != ('lower' if scale == 'error' else 'upper'):
        raise ValueError('所选准确率/错误率与日志的优化方向不一致。')
    rows = []
    configs = run.get_configs()
    for i,t in enumerate(sorted(run.get_trials(),key=lambda t:(t.end_time,t.start_time)),1):
        if t.status != Status.SUCCESS or t.costs[objective] is None:
            continue
        value = float(t.costs[objective])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('要求 0–1 的准确率或错误率，不能把任意 cost 当作错误率。')
        rows.append((i,1-value if scale == 'error' else value,dict(configs[t.config_id])))
    if not rows:
        raise ValueError('没有成功试验。')
    return rows

def importance(rows):
    if len(rows)<10 or np.ptp([r[1] for r in rows])<1e-12:
        raise ValueError('至少需要 10 条且成绩存在变化的成功试验。')
    frame=pd.DataFrame([r[2] for r in rows]); blocks=[]; owners=[]
    for name in frame:
        col=frame[name]
        if pd.api.types.is_numeric_dtype(col):
            block=col.fillna(col.median() if col.notna().any() else 0).to_numpy().reshape(-1,1)
            if col.isna().any():
                block=np.column_stack([block,col.isna().astype(int)])
        else:
            if col.nunique(dropna=False)>128:
                raise ValueError('单个类别参数超过 128 个取值，无法进行当前重要性分析。')
            block=pd.get_dummies(col.fillna('(inactive)').astype(str)).to_numpy()
        blocks.append(block); owners.extend([name]*block.shape[1])
        if len(owners)>512:
            raise ValueError('参数编码超过 512 维，请减少类别数量或参数。')
    if not blocks:
        raise ValueError('缺少超参数配置。')
    model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1)
    model.fit(np.column_stack(blocks),[r[1] for r in rows])
    scores={name:0. for name in frame}
    for name,value in zip(owners,model.feature_importances_): scores[name]+=float(value)
    return dict(sorted(scores.items(),key=lambda item:item[1],reverse=True))

class GenericComparison(Plugin):
    id='deepcave_x_trajectory'
    name='轨迹对比与参数重要性'
    icon='fas fa-chart-line'

    def register_callbacks(self):
        from deepcave import app
        for side in ['a','b']:
            def register(side):
                @app.callback([Output('obj-'+side,'options'),Output('obj-'+side,'value')],Input('run-'+side,'value'))
                def objectives(path):
                    run=runs().get(path)
                    options=[] if run is None else [{'label':o.name+' ('+o.optimize+')','value':i}
                        for i,o in enumerate(run.get_objectives()) if o.name!='Time']
                    return options,options[0]['value'] if options else None
            register(side)
        @app.callback([Output('generic-curve','figure'),Output('generic-importance','figure'),Output('generic-note','children')],
            [Input('run-a','value'),Input('run-b','value'),Input('obj-a','value'),Input('obj-b','value'),
             Input('scale-a','value'),Input('scale-b','value'),Input('confirm-comparison','value'),Input('importance-run','value')])
        def update(a,b,oa,ob,sa,sb,confirmed,selected):
            if 'yes' not in (confirmed or []):
                return go.Figure(),go.Figure(),'请选择日志并确认实验条件。'
            try:
                registry=runs(); rows=[extract(registry[a],oa,sa),extract(registry[b],ob,sb)]
                fig=go.Figure()
                for label,rr in zip(['A','B'],rows):
                    fig.add_trace(go.Scatter(x=[r[0] for r in rr],y=np.maximum.accumulate([r[1] for r in rr]),
                        name=label+' 最佳成绩',mode='lines+markers',line_shape='hv'))
                    fig.add_trace(go.Scatter(x=[r[0] for r in rr],y=[r[1] for r in rr],name=label+' 单次成绩',
                        mode='markers',text=[str(r[2]) for r in rr],hovertemplate='%{y:.4%}<br>%{text}<extra></extra>',visible='legendonly'))
                fig.update_layout(template='plotly_white',height=440,xaxis_title='已完成试验次数（从 1 开始）',yaxis_title='准确率',yaxis_tickformat='.1%')
                note='；'.join(f'{label}：{len(rr)} 条成功试验，最佳准确率 {max(r[1] for r in rr):.4%}' for label,rr in zip(['A','B'],rows))
                imp=go.Figure()
                try:
                    scores=importance(rows[int(selected)])
                    imp.add_trace(go.Bar(x=list(scores),y=list(scores.values())))
                    imp.update_layout(template='plotly_white',title='运行 '+('A' if selected=='0' else 'B')+'：超参数重要性',height=400,yaxis_title='归一化重要性')
                except ValueError as exc: note+='。重要性分析：'+str(exc)
                return fig,imp,note
            except Exception as exc: return go.Figure(),go.Figure(),'无法比较：'+str(exc)

    def __call__(self,render_button=False):
        options=[{'label':path,'value':path} for path in runs()]
        controls=[]
        for side in ['a','b']:
            controls.extend([html.H3('运行 '+side.upper()),dcc.Dropdown(id='run-'+side,options=options,placeholder='选择已加载的日志',clearable=False),
                dcc.Dropdown(id='obj-'+side,options=[],placeholder='目标指标',clearable=False),
                dcc.Dropdown(id='scale-'+side,options=[{'label':'准确率（越高越好）','value':'accuracy'},
                    {'label':'错误率 = 1 − 准确率','value':'error'}],value='accuracy',clearable=False)])
        return [html.H1('DeepCAVE-X：轨迹对比与参数重要性'),html.P('先在首页加载新的日志文件夹，再进入本页选择。支持任意已加载运行的 0–1 准确率或错误率。'),*controls,
            dcc.Checklist(id='confirm-comparison',options=[{'label':' 我已确认两组实验的数据集、划分、指标和预算一致','value':'yes'}],value=[]),
            dcc.Graph(id='generic-curve'),html.Div(id='generic-note'),html.H2('超参数重要性'),
            dcc.RadioItems(id='importance-run',options=[{'label':'运行 A','value':'0'},{'label':'运行 B','value':'1'}],value='0'),dcc.Graph(id='generic-importance'),
            html.P('随机森林回归拟合“超参数→成绩”，以不纯度减少量（MDI）估计重要性；类别编码合并回原参数。不是 Sobol 指数或因果贡献。结果受搜索范围和采样影响，20–30 次试验仅作探索性分析。点击图例可显示单次成绩。试验次数不等于运行时间。')]
