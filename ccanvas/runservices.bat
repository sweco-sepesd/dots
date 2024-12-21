@echo off
setlocal
pushd %~dp0
call env.bat
start /b nginx
endlocal
popd
echo on
