@echo off
chcp 936 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title graph-rag-knowledge-base  START DOCKER SERVICES

rem --no-pause 供脚本/计划任务调用，避免末尾等待按键
set "NOPAUSE="
if /i "%~1"=="--no-pause" set "NOPAUSE=1"

set "COMPOSE_FILE=%~dp0docker-compose-milvus.yml"
rem 已有容器叫 minio；若走 compose 新建则是 milvus-minio
set "MINIO_CNAME=minio"

echo ============================================================
echo   Graph RAG 依赖服务一键启动 (Docker)
echo ------------------------------------------------------------
echo   etcd      :2379                Milvus 元数据
echo   MinIO     :9000 / :9001       对象存储
echo   Milvus    :19530              向量库
echo   Neo4j     :7474 / :7687       知识图谱
echo   MongoDB   :27017              会话历史
echo ============================================================
echo.

rem ---------------- 1. 确认 docker 命令 ----------------
where docker >nul 2>&1
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;!PATH!"
  )
)
where docker >nul 2>&1
if errorlevel 1 (
  echo [FAIL] 未找到 docker 命令，请先安装 Docker Desktop。
  goto :fail
)

rem ---------------- 2. 确认 Docker 引擎 ----------------
docker info >nul 2>&1
if not errorlevel 1 (
  echo [OK] Docker 引擎已在运行
  goto :engine_ready
)

set "DOCKER_DESKTOP=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
if not exist "!DOCKER_DESKTOP!" set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "!DOCKER_DESKTOP!" (
  echo [FAIL] Docker 引擎未运行，且未找到 Docker Desktop，请手动启动 Docker 后重试。
  goto :fail
)

echo [..] Docker 引擎未运行，正在启动 Docker Desktop（最多等待 180 秒）...
start "" "!DOCKER_DESKTOP!"
call :wait_docker 180
if errorlevel 1 (
  echo [FAIL] 等待 Docker 引擎启动超时。
  goto :fail
)
echo [OK] Docker 引擎已就绪

:engine_ready

rem ---------------- 3. Milvus 容器组 ----------------
set "MILVUS_STACK_OK=1"
for %%C in (milvus-etcd minio milvus-standalone) do (
  docker inspect %%C >nul 2>&1 || set "MILVUS_STACK_OK=0"
)

if "!MILVUS_STACK_OK!"=="1" (
  echo.
  echo [1/3] 启动 etcd 与 MinIO ...
  docker start milvus-etcd minio >nul
  call :wait_port 2379 60
  call :wait_port 9000 90

  echo [2/3] 启动 Milvus standalone ...
  docker start milvus-standalone >nul
  call :wait_port 19530 180
) else (
  echo.
  echo [1/3] 未找到完整的 Milvus 容器组，改用 docker compose 创建 ...
  docker compose -f "%COMPOSE_FILE%" up -d
  if errorlevel 1 (
    echo [FAIL] docker compose 启动失败。
    goto :fail
  )
  set "MINIO_CNAME=milvus-minio"
  call :wait_port 19530 180
)

rem ---------------- 4. Neo4j / MongoDB ----------------
echo.
echo [3/3] 启动 Neo4j 与 MongoDB ...

docker inspect neo4j >nul 2>&1
if errorlevel 1 (
  echo    [..] 未找到 neo4j 容器，按默认配置创建 ...
  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/12345678 -v neo4j_data:/data -v neo4j_logs:/logs neo4j:5
) else (
  docker start neo4j >nul
)
call :wait_port 7687 90

docker inspect mongo >nul 2>&1
if errorlevel 1 (
  echo    [..] 未找到 mongo 容器，按默认配置创建 ...
  docker run -d --name mongo -p 27017:27017 -v mongo_data:/data/db -v mongo_config:/data/configdb mongo:6.0
) else (
  docker start mongo >nul
)
call :wait_port 27017 60

rem ---------------- 5. 汇总 ----------------
echo.
echo ============================================================
echo   容器状态
echo ------------------------------------------------------------
docker ps -a --filter "name=milvus-etcd" --filter "name=milvus-standalone" --filter "name=minio" --filter "name=neo4j" --filter "name=mongo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ============================================================

set "ALL_OK=1"
for %%C in (milvus-etcd milvus-standalone neo4j mongo) do (
  docker inspect -f "{{.State.Status}}" %%C 2>nul | findstr /c:"running" >nul || set "ALL_OK=0"
)
docker inspect -f "{{.State.Status}}" !MINIO_CNAME! 2>nul | findstr /c:"running" >nul || set "ALL_OK=0"

if "!ALL_OK!"=="1" (
  echo [OK] 全部依赖服务已启动。
  echo.
  echo 下一步：双击 start_import_8000.bat / start_query_8001.bat
  echo 体检：  F:\Program\miniconda\envs\graph-rag\python.exe scripts\health_check.py --no-services
  echo.
  echo 提示：这些容器的重启策略为 no，电脑重启后需再次运行本脚本。
  if not defined NOPAUSE (
    echo.
    pause
  )
  exit /b 0
) else (
  echo [WARN] 有容器未处于 running 状态，查看日志：
  echo        docker logs --tail 100 ^<容器名^>
  if not defined NOPAUSE (
    echo.
    pause
  )
  exit /b 1
)

:fail
echo.
echo [FAILED] 依赖服务未能全部启动，请查看上面的输出。
if not defined NOPAUSE (
  echo.
  pause
)
exit /b 1

rem ==================== subroutines ====================

:wait_docker
rem  %1 = timeout seconds
set /a _wd=0
:wait_docker_loop
docker info >nul 2>&1
if not errorlevel 1 exit /b 0
set /a _wd+=3
if !_wd! geq %~1 exit /b 1
ping -n 4 127.0.0.1 >nul 2>&1
goto :wait_docker_loop

:wait_port
rem  %1 = port, %2 = timeout seconds
set /a _wp=0
:wait_port_loop
netstat -ano | findstr /c:":%~1 " | findstr /c:"LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo    [OK] 端口 %~1 已就绪
  exit /b 0
)
set /a _wp+=2
if !_wp! geq %~2 (
  echo    [WARN] 等待端口 %~1 超时（%~2 秒）
  exit /b 1
)
ping -n 3 127.0.0.1 >nul 2>&1
goto :wait_port_loop
