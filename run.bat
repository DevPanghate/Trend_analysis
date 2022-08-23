@echo off

:start
set /p Loc=Input_File_Location?
set /p r=DataFile?
pause
cd %Location%
if EXIST %r% ( 
py channel_formation.py -l %r%
pause
cls
goto start
) else (
py channel_formation.py -l inputfile.txt
pause
cls
goto start
)

