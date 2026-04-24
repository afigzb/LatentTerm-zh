@echo off
REM ============================================================
REM  LatentTerm 潜词  ·  启动脚本
REM  双击本文件即可启动，关闭窗口即退出程序。
REM ============================================================

chcp 65001 > nul
title LatentTerm 潜词 - 启动中
cd /d "%~dp0"

echo.
echo ============================================================
echo           LatentTerm 潜词 - 中文小说术语挖掘
echo ============================================================
echo.

if not exist "%~dp0python\python.exe" (
    echo [错误] 未找到 Python 运行时。
    echo        请确认本目录下存在 python\ 子目录。
    echo        可能是解压不完整，请重新解压压缩包后再试。
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0libs" (
    echo [错误] 未找到 libs\ 依赖目录。
    echo        可能是解压不完整，请重新解压压缩包后再试。
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0app.py" (
    echo [错误] 未找到 app.py 主程序。
    echo.
    pause
    exit /b 1
)

echo 正在启动服务，首次启动约需 5~15 秒...
echo 启动完成后浏览器会自动打开。
echo 如浏览器未自动打开，请手动访问：  http://localhost:8965
echo.
echo 关闭本窗口即可退出程序。
echo ============================================================
echo.

set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set STREAMLIT_CREDENTIALS_EMAIL=none@example.com
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

"%~dp0python\python.exe" -m streamlit run "%~dp0app.py" ^
    --server.port=8965 ^
    --server.headless=false ^
    --server.fileWatcherType=none ^
    --browser.gatherUsageStats=false

set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo ============================================================
    echo [异常退出] 退出码: %EXITCODE%
    echo 请截图本窗口内容反馈给开发者。
    echo ============================================================
    pause
)

exit /b %EXITCODE%
