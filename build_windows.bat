@echo off

echo ===============================================
echo   CEST-MRF GUI  -- Windows build
echo ===============================================

echo.
echo Cleaning previous build artefacts...
if exist build\     rmdir /s /q build
if exist dist\CEST_MRF_GUI  rmdir /s /q dist\CEST_MRF_GUI
if exist dist\CEST_MRF_GUI.exe  del dist\CEST_MRF_GUI.exe

echo.
echo Running PyInstaller...
pyinstaller CEST_MRF_GUI.spec --clean --noconfirm --log-level WARN

if not exist dist\CEST_MRF_GUI (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)
echo    OK: dist\CEST_MRF_GUI created.

set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist %ISCC% (
    echo.
    echo Building installer with Inno Setup...
    %ISCC% CEST_MRF_GUI_installer.iss
    echo    OK: Installer created.
) else (
    echo.
    echo Inno Setup not found -- skipping installer creation.
    echo To create an installer, download Inno Setup from:
    echo   https://jrsoftware.org/isinfo.php
    echo Then re-run this script.
)

echo.
echo ===============================================
echo   DONE!
echo   Distributable folder: dist\CEST_MRF_GUI\
echo   Share the entire folder or zip it.
echo ===============================================
pause
