@echo off
setlocal

cd /d %~dp0

echo [SmartPlace] 正在配置主环境 .venv ...

where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 python，请先安装 Python 3.12 并加入 PATH。
    pause
    exit /b 1
)

if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo 创建 .venv 失败。
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo 激活 .venv 失败。
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo pip 升级失败。
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo 主环境依赖安装失败。
    pause
    exit /b 1
)

echo.
echo [SmartPlace] 主环境安装完成。
echo 可继续运行 setup_libcom.bat 安装 libcom 子环境。
pause
