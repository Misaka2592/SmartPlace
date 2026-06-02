@echo off
setlocal

cd /d %~dp0

if not exist .venv\Scripts\python.exe (
    echo 未找到主环境 .venv 。
    echo 请先运行 setup_main.bat
    pause
    exit /b 1
)

if not exist .venv_libcom\Scripts\python.exe (
    echo 未找到 libcom 子环境 .venv_libcom 。
    echo 请先运行 setup_libcom.bat
    pause
    exit /b 1
)

echo [SmartPlace] 正在启动应用 ...
echo 浏览器打开地址：http://127.0.0.1:7860
echo.

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo 激活主环境失败。
    pause
    exit /b 1
)

python app.py

echo.
echo 应用已退出。
pause
