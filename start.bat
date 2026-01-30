@echo off
chcp 65001 >nul
title K線形態辨識系統 v2.0

cd /d "%~dp0"
uv run python main.py %*

pause
