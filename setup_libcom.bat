@echo off
setlocal

cd /d %~dp0

echo [SmartPlace] 正在配置 libcom 子环境 .venv_libcom ...

where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 python，请先安装 Python 3.12 并加入 PATH。
    pause
    exit /b 1
)

if not exist third_party\libcom\setup.py (
    echo 未找到 third_party\libcom 。
    echo 请先执行：
    echo git clone https://github.com/bcmi/libcom.git third_party/libcom
    pause
    exit /b 1
)

if not exist .venv_libcom (
    python -m venv .venv_libcom
    if errorlevel 1 (
        echo 创建 .venv_libcom 失败。
        pause
        exit /b 1
    )
)

call .venv_libcom\Scripts\activate.bat
if errorlevel 1 (
    echo 激活 .venv_libcom 失败。
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo pip 升级失败。
    pause
    exit /b 1
)

pip install -r requirements_libcom.txt
if errorlevel 1 (
    echo libcom 子环境依赖安装失败。
    pause
    exit /b 1
)

echo.
echo [SmartPlace] libcom 子环境安装完成。
echo 现在可以运行 run_app.bat 启动项目。
pause
