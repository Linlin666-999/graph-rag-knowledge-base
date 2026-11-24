import os
import subprocess
import sys
import shutil
import time
import json
import socket
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError,FileProcessingError,PdfConversionError

class Pdf_To_Md_Node(BaseNode):
    """"
    pdf转node节点
    """
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """
        #1. 参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        #2. 利用MinerU工具解析pdf->md(4.x: -o 是输出文件, 需要显式指定全页与解析档位)
        md_path = self._get_md_paths(import_file_path, file_dir_path)
        processed_code = self._execute_mineru(import_file_path, Path(md_path))
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        #3. 校验md确实生成(4.x 服务端失败时可能仍返回0, 不能只靠退出码)
        md_path_obj = Path(md_path)
        if not md_path_obj.is_file() or md_path_obj.stat().st_size == 0:
            raise PdfConversionError(
                f"MinerU未生成有效的Markdown文件: {md_path}", self.name
            )

        #4. 更新state 字典的md_path
        state["md_path"] = md_path

        #5. 返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        Args:
            state:该节点接收到的状态

        Returns:

        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf文件路径
        import_file_path = state.get("import_file_path", "")
        # 2. 获取解析后的输出目录
        file_dir = state.get("file_dir", "")
        # 3. 校验输入的文件路径
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)
        # 4. 用Path标准化
        import_file_path_opj = Path(import_file_path)

        # 5. 校验是否是一个真实的路径
        if not import_file_path_opj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        # 6. 判断输出目录是否为空
        if not file_dir:
            # 默认目录兜底
            file_dir = import_file_path_opj.parent  # 只有经历过标准化的路径才能拥有parent

        # 7. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8. 返回 输出文件以及输出目录的标准path
        return import_file_path_opj, file_dir_path_obj

    @staticmethod
    def _resolve_mineru_exe() -> str:
        """
        定位 mineru 可执行文件。Windows 上 conda 环境的 Scripts 目录不在 PATH 中,
        直接用 "mineru" 会让 subprocess.Popen 抛 WinError 2。
        优先用当前 Python 解释器同环境的 Scripts/mineru.exe, 其次回退到 PATH。
        """
        env_exe = Path(sys.executable).parent / "Scripts" / "mineru.exe"
        if env_exe.is_file():
            return str(env_exe)
        found = shutil.which("mineru")
        if found:
            return found
        raise PdfConversionError(
            f"找不到 mineru 可执行文件(已检查 {env_exe} 与 PATH)",
            Pdf_To_Md_Node.name,
        )

    @staticmethod
    def _build_child_env() -> dict:
        """
        4.x 通过 127.0.0.1 上的 doclib 服务通信; httpx 默认 trust_env=True,
        本机开着 Clash 之类的系统代理时, 本地请求会被交给代理导致连接失败。
        因此显式清空代理变量并设置 NO_PROXY 让本地地址直连。
        """
        child_env = os.environ.copy()
        child_env["NO_PROXY"] = "127.0.0.1,localhost,::1"
        child_env["no_proxy"] = "127.0.0.1,localhost,::1"
        for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                          "http_proxy", "https_proxy", "all_proxy"):
            child_env.pop(proxy_key, None)
        return child_env

    @staticmethod
    def _parse_server_url() -> str:
        """从 doclib.endpoint.json 读取本地服务地址(4.x 把端点写在此文件)。"""
        endpoint_file = Path.home() / ".mineru" / "doclib.endpoint.json"
        try:
            data = json.loads(endpoint_file.read_text(encoding="utf-8"))
            for transport in data.get("transports", []):
                if transport.get("type") == "tcp" and transport.get("base_url"):
                    return transport["base_url"]
        except (OSError, ValueError, KeyError):
            pass
        return "http://127.0.0.1:15980"

    @staticmethod
    def _server_is_up(url: str) -> bool:
        """
        用 socket 直接探测 TCP 端点。`mineru server status` 每次要加载 doclib
        客户端, 实测约 35s, 放在每个上传任务里会超时; 而端口探测只要毫秒级。
        HTTP 401(需 token)/404 都说明服务在监听, 只有连接失败才算未启动。
        """
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    def _ensure_mineru_server(self, mineru_exe: str, child_env: dict) -> None:
        """
        4.x 的 parse 依赖本地 doclib 服务。若服务未运行, 先执行 `mineru server start`
        (幂等, 已在运行时会直接返回), 再轮询端点等待就绪。
        """
        server_url = self._parse_server_url()
        if self._server_is_up(server_url):
            return

        self.logger.info("MinerU 本地服务未运行, 执行 mineru server start ...")
        start_proc = subprocess.run(
            [mineru_exe, "server", "start"],
            env=child_env, capture_output=True, text=True,
            errors="replace", timeout=60,
        )
        self.logger.info(f"server start 输出: {start_proc.stdout.strip()}")

        # 等待服务就绪(首次启动需初始化 doclib / 加载模型)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self._server_is_up(server_url):
                self.logger.info("MinerU 本地服务已就绪")
                return
            time.sleep(3)
        raise PdfConversionError(
            "MinerU 本地服务启动超时(120s), 请手动执行 mineru server status 排查",
            self.name,
        )

    def _execute_mineru(self, import_file_path: Path, md_path: Path) -> int:
        """
        执行 MinerU 4.x 的 parse 子命令解析 PDF -> Markdown。

        与 2.x 的差异(2.x 的 `mineru -p <pdf> -o <dir> --source local` 已失效):
          1. 4.x 用 `mineru parse <path>`, `-p` 变成了页码选项 `--pages`;
          2. `-o` 是**输出文件路径**, 不再是输出目录, 且会自动创建父目录;
          3. `--pages` 默认只解析前 10 页, 必须显式传 `all` 才能覆盖整本;
          4. 4.x 是客户端/服务端架构, 需要本地 `mineru server` 在跑;
             没有 basic/standard 解析引擎时, 需显式 `--tier flash`(纯文本档位),
             否则报 "No basic, standard, or advanced engine available"。

        Args:
            import_file_path: 待解析的 PDF 路径
            md_path: 输出的 .md 文件路径

        Returns:
            MinerU 子进程的退出码
        """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 定位 mineru 可执行文件并确保 4.x 的本地服务已启动
        mineru_exe = self._resolve_mineru_exe()
        child_env = self._build_child_env()
        self._ensure_mineru_server(mineru_exe, child_env)

        # 2. 构建命令行; 档位可用环境变量 MINERU_TIER 覆盖, 默认 flash(本机唯一可用档位)
        tier = os.getenv("MINERU_TIER", "flash").strip() or "flash"
        # flash 档位解析文本型 PDF 很快, 但扫描件会走本地 OCR 模型, 常超过
        # CLI 默认的 60 秒等待; 默认放宽到 600 秒, 可用 MINERU_WAIT 覆盖。
        wait_seconds = os.getenv("MINERU_WAIT", "600").strip() or "600"
        cmd = [
            mineru_exe,
            "parse",
            str(import_file_path),
            "--tier", tier,
            "--pages", "all",
            "--wait", wait_seconds,
            "-f", "markdown",
            "-o", str(md_path),
        ]

        process_start_time = time.time()

        self.logger.info(f"MinerU命令: {' '.join(cmd)}")

        # 3.执行子进程, 子进程能够自动读取到主进程的环境变量
        proc = subprocess.Popen(
            args=cmd,
            env=child_env,
            stdout=subprocess.PIPE,  # 主进程开一根管子,专门用来读取子进程的输出(错误日志+正常日志都从这根管子出来)
            stderr=subprocess.STDOUT,  # 把子进程的错误日志并到上面那根 stdout 管子里,两者合并成一路输出
            errors="replace",  # 保证遇到一些错误没办法解释的时候不会崩,用一些会的内容进行替换   比如,输出 ?
            text=True,  # 输出字符串
            encoding="utf-8",
            bufsize=1  # 按行缓存区, 只要一行满了就换行
        )

        # 4. 获取日志信息, stdout是流式产生,所以需要迭代输出
        for outlog in proc.stdout:
            self.logger.info(f"执行Mineru产生的日志:{outlog.rstrip()}")

        # 5. 等待子进程处理完(主进程等待子进程做完
        # ),做完会返回处理码0,如果不是0就没有处理完
        processed_code = proc.wait()

        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(f"MinerU成功解析PDF文件:{import_file_path.name}, 耗时:{process_end_time - process_start_time:.2f}") #因为路径经过path标准化之后就可以调用.name属性
        else:
            self.logger.error(f"MinerU解析PDF文件失败: {import_file_path.name}, 退出码:{processed_code}")
        # 5. 返回状态码
        return processed_code

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path) -> str:
        """
        4.x 的 -o 是单个输出文件, 不再是 2.x 的 `<dir>/<stem>/hybrid_auto/<stem>.md`
        目录结构。这里直接把 md 输出到上传文件所在目录, 保持后续 md_img_node
        对 md_path.exists() 的读取要求, 且 images 目录会落在同级(不存在即跳过图片处理)。
        """
        return str(file_dir_path / f"{import_file_path.stem}.md")







if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = Pdf_To_Md_Node()

    test_entry_state = {
        "file_dir": r"F:\agent_project_knowledgestore\shopkeeper_brain\knowledge\processor\import_process",
        "import_file_path": r"C:\Users\尤锦滨\Desktop\面向AI Agent的零信任安全机制设计.pdf"
    }

    pdf_to_md_node(test_entry_state)