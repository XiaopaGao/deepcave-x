# DeepCAVE-X

上传 Optuna 或 SMAC 超参数优化日志，进行轨迹对比、随机森林 MDI 超参数重要性分析和停滞区间诊断。

## 支持的日志

- Optuna CSV：`study.trials_dataframe().to_csv("optuna.csv", index=False)`
- Optuna JSON：使用网页提供的 `assets/export_optuna.py`
- SMAC 2.x ZIP：包含 `configspace.json`、`scenario.json`、`runhistory.json`

网站不要求上传原始数据集或模型文件，也不接受 pickle 文件。上传内容临时保存在单个服务进程的内存中；连续两小时无访问或服务重启后失效。

## 本地启动

建议使用 Python 3.9：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

打开 `http://127.0.0.1:8051/`。

## Render 部署

完整步骤见 `部署到Render步骤.md`。仓库根目录已经包含 `render.yaml`。生产启动命令必须保持一个 worker，因为当前访问者工作区保存在进程内存中。

## 方法边界

- 重要性采用随机森林 MDI，不代表因果影响，也不是 Sobol 指数。
- 停滞诊断基于 patience 和 min_delta，是对已有日志的阈值分析，不是实时训练监控。
- 轨迹按当前预算下的成功试验完成顺序展示，试验次数不等于运行时间。
- 比较前须人工核对数据集、数据划分、指标、搜索空间和预算。

DeepCAVE 项目：https://github.com/automl/DeepCAVE

