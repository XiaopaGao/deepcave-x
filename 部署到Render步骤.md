# DeepCAVE-X 部署到 Render

## 一、把本文件夹上传到 GitHub

1. 登录 GitHub，点击右上角 `+`，选择 `New repository`。
2. 仓库名填写 `deepcave-x`，可选择 Private。
3. 创建仓库后，把本文件夹中的所有文件上传到仓库根目录。
4. 确认 GitHub 首页能直接看到 `app.py`、`requirements.txt` 和 `render.yaml`，不要在仓库中再套一层文件夹。
5. 不要上传实验日志、数据集、`.pkl`、`.env` 或密钥。

## 二、在 Render 创建服务

推荐使用 Blueprint，因为本项目已经包含 `render.yaml`：

1. 登录 https://render.com 并连接 GitHub。
2. 在 Dashboard 中选择 `New` → `Blueprint`。
3. 选择刚创建的 `deepcave-x` 仓库。
4. Render 读取 `render.yaml` 后会显示一个名为 `deepcave-x` 的 Web Service。
5. 确认后开始部署。

如果不用 Blueprint，而是选择 `New` → `Web Service`，填写：

- Language：`Python 3`
- Build Command：`pip install -r requirements.txt`
- Start Command：`gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
- Health Check Path：`/`

在 Environment 中增加 `DEEPCAVE_SECRET`，选择 Generate 自动生成。不要把密钥写入 GitHub。

## 三、等待和检查

部署日志中应先完成依赖安装，然后出现服务启动成功。最终网址类似：

`https://deepcave-x.onrender.com`

依次检查：

1. 首页可以打开。
2. Optuna CSV 可以上传并出现在运行 A。
3. SMAC ZIP 可以上传并出现在运行 B。
4. 轨迹叠加和并排视图均能显示。
5. 重要性和瓶颈诊断能显示。
6. 用无痕窗口打开网站，看不到普通窗口上传的运行。

## 四、免费版的表现

免费 Web Service 空闲后会休眠，下一次访问可能等待约一分钟，但固定 `onrender.com` 地址不会像临时 Cloudflare 地址那样每次变化。若需要持续快速响应，在 Render 中把此 Web Service 升级为付费实例。

当前上传日志保存在内存中，不需要数据库或持久硬盘。服务重启、重新部署或免费实例休眠后，用户需要重新上传日志，这符合临时分析用途。

## 五、更新网站

修改代码后提交到 GitHub 的 `main` 分支，Render 会自动重新构建和部署。构建失败时查看 Render 的 Deploy Logs，修正后再次提交。

