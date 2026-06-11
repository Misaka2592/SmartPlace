# SmartPlace

SmartPlace 是一个本地运行的前景放置推荐与合成图像分析系统。  
用户可以选择或上传前景图和背景图，在画布中拖拽前景物体到多个候选位置，系统会调用真实 `libcom` OPA 模型与 SmartPlace 校准层，对候选位置进行评分、排序、解释和导出。

## 1. 项目内容

- `app.py`：主应用入口，启动本地 Gradio 网页
- `assets/foregrounds/`：前景预制图片
- `assets/backgrounds/`：背景预制图片
- `models/`：评分模型、`libcom` 子进程封装、SmartPlace 校准模型
- `utils/`：图像合成、评分、解释图、导出、日志、案例汇总
- `scripts/`：`libcom` 推理脚本、MindSpore 演示脚本、CANN/昇腾演示脚本
- `configs/default.yaml`：运行配置

## 2. 环境说明

建议环境：

- Windows 10 或 Windows 11
- Python 3.12
- 有 NVIDIA GPU 更好，没有 GPU 也能跑，但 `libcom` 相关推理会慢很多

本项目建议使用两个 Python 虚拟环境：

1. `.venv`
   用于主应用，也就是 `app.py` 和 Gradio 网页。

2. `.venv_libcom`
   用于子进程调用真实 `libcom` 模型，避免主环境和 `libcom` 依赖冲突。

可选：

3. `.venv_ms`
   用于 MindSpore / CANN 相关脚本，不是本地网页运行的必需环境。

## 3. 获取项目代码

先克隆 SmartPlace 仓库：

```bash
git clone <你的 SmartPlace 仓库地址>
cd SmartPlace
```

注意：

`third_party/libcom/` 当前没有提交到仓库，因为它被 `.gitignore` 忽略了。  
所以每个组员都需要额外把 `libcom` 克隆到 `third_party/libcom`：

```bash
git clone https://github.com/bcmi/libcom.git third_party/libcom
```

## 4. 安装主环境

手动安装方式：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

也可以直接双击或在终端运行：

```bash
setup_main.bat
```

## 5. 安装 libcom 环境

手动安装方式：

```bash
python -m venv .venv_libcom
.venv_libcom\Scripts\activate
pip install --upgrade pip
pip install -r requirements_libcom.txt
```

也可以直接双击或在终端运行：

```bash
setup_libcom.bat
```

`requirements_libcom.txt` 的前提是：

- `third_party/libcom/` 已经存在
- 会先安装 `third_party/libcom/requirements.txt` 里的依赖
- 再以可编辑模式安装本地 `libcom`

## 6. 启动项目

手动启动方式：

```bash
.venv\Scripts\activate
python app.py
```

也可以直接双击或在终端运行：

```bash
run_app.bat
```

启动成功后，在浏览器打开：

```text
http://127.0.0.1:7860
```

## 7. 基本复现流程

1. 打开 `http://127.0.0.1:7860`
2. 选择前景预制图，或者上传自己的前景图
3. 选择背景预制图，或者上传自己的背景图
4. 点击加载画布
5. 在画布中拖拽前景物体
6. 记录多个候选位置
7. 点击批量评分并生成结果
8. 查看：
   `结果仪表盘`
   `解释与案例库`
   `导出与复现`

## 8. 预制素材说明

工作台中的预制图片会自动读取以下目录中的前 6 张图片：

- `assets/foregrounds/`
- `assets/backgrounds/`

如果替换或新增这两个目录中的图片，重启应用后，工作台中的预制图会自动更新。

## 9. MindSpore / CANN 说明

这部分不是本地网页运行的必需条件，但属于项目的扩展交付内容。

MindSpore 演示：

```bash
python -m venv .venv_ms
.venv_ms\Scripts\activate
pip install -r requirements_mindspore.txt
python scripts/run_mindspore_demo.py
```

CANN / 昇腾演示：

- 建议在华为云 ModelArts / Ascend Notebook 环境中运行
- 使用脚本：`scripts/run_cann_ascend_demo.py`
- 输出结果通常在 `outputs/cann/`

## 10. 常见问题

`third_party/libcom` 不存在：

```text
libcom 子环境无法导入 libcom。
解决方法：先执行
git clone https://github.com/bcmi/libcom.git third_party/libcom
```

`7860` 端口被占用：

```text
先关闭旧的 Python 进程，再重新运行 run_app.bat 或 python app.py。
```

推理速度很慢：

```text
没有 GPU 时，libcom 推理会明显变慢。
开启解释图和 LibCom 增强模型时，会比普通评分更慢。
```

网页里看不到预制图片：

```text
检查 assets/foregrounds/ 和 assets/backgrounds/ 中是否真的有图片文件。
```