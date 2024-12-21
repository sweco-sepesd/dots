@echo off
setlocal
pushd %~dp0
call env.bat
nginx -s stop
endlocal
popd
echo on