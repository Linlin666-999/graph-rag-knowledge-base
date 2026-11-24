import uuid
import shutil
from fastapi import UploadFile, HTTPException
from knowledge.processor.import_process.core.paths import get_local_base_dir
from datetime import datetime
import os.path

from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.minio_util import get_minio_client
from knowledge.processor.import_process.services.task_service import TaskService
from knowledge.processor.import_process.nodes.main_graph import kb_import_graph_app

class ImportFileService:
    """
    文件导入的业务类

    1. 保存上传的文件到本地
    2. 保存上传的文件到minio
    3. 运行图谱的所有节点
    """

    def __init__(self, task_service: TaskService):
        self._task_service = task_service

    def get_date_dir(self) ->str:

        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))    #拼接目录

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        """还原 multipart 中文文件名。

        Starlette 按 latin-1 解码 multipart 的 filename，而客户端实际发送的是
        原始字节：浏览器通常是 UTF-8，Windows 下的 curl/终端可能是 GBK。
        这里先 latin-1 编码拿回原始字节，再依次尝试 UTF-8、GB18030 解码；
        不是乱码（例如已经是正常中文）时原样返回。
        """
        if not filename:
            return filename
        try:
            raw = filename.encode("latin-1")
        except UnicodeEncodeError:
            # 含 U+00FF 以上字符，说明已经是正常解码的文本
            return filename

        for encoding in ("utf-8", "gb18030"):
            try:
                restored = raw.decode(encoding)
                # 纯 ASCII 没必要改动；解码结果与原文不同才认为还原成功
                if restored != filename:
                    return restored
            except UnicodeDecodeError:
                continue
        return filename

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str):
        """
        将上传的文件保存到本地
        Args:
            file:
            file_dir:

        Returns:

        """
        # 1. 确保归档目录存在
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建上传文件的完整的path
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 写入(批量的写入shutil.copyfileobj(file.file,f))
        with open(import_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # 4. 返回导入文件的path
        return import_file_path


    def save_upload_file_to_minio(self, import_file_path: str, file: UploadFile):
        """
        Args:
            import_file_path:
            file:

        Returns:

        """
        # 1. 获取minio客户端
        minio_client = get_minio_client()

        # 2. 判断minio客户端是否存在
        if not minio_client:
            raise HTTPException(status_code=500, detail="MinIO service not available")

        # 3. 构建Minio客户端对象名
        minio_object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"

        # 4. 获取桶名
        bucket_name = os.getenv("MINIO_BUCKET_NAME")

        # 5. 开始上传
        try:
            minio_client.fput_object(bucket_name, minio_object_name, import_file_path)
        except Exception as e:
            raise ValueError(f"{file.filename}文件上传失败 原因:{e}")

        return minio_object_name

    def process_upload_file(self,file: UploadFile):
        """
        处理文件上传
        Returns:
        1. 标记当前文件上传节点('upload_file')为正在运行中
        2. 将上传的文件保存在本地
        3. 将上传的文件保存在minio
        4. 标记当前文件上传节点运行完毕
        5. 需要返回三部分信息(task_id, file_dir, import_file_path)
        """
        # 1. 构建时间日期的文件目录出来
        date_dir = self.get_date_dir()

        # 2. 生成一个任务id
        task_id = str(uuid.uuid4())

        # 2.1 还原中文文件名（Starlette 默认按 latin-1 解码 multipart filename）
        if file.filename:
            file.filename = self._normalize_filename(file.filename)

        # 3. 构建文件的最终归属目录
        file_dir = os.path.join(date_dir, task_id)

        self._task_service.mark_node_running(task_id, "upload_file")

        # 4. 将接收到的文件上传本地
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        #5.将本地磁盘的上传文件同步到minio中
        self.save_upload_file_to_minio(import_file_path,file)
        self._task_service.mark_node_done(task_id, "upload_file")

        #6. 构建返回值
        return task_id, file_dir, import_file_path

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """
        运行导入graph的流程（跑节点）
        Args:
            task_id:
            file_dir:
            import_file_path:

        Returns:

        """

        # 1. 构建初始状态
        # graph.stream()之前调用update_task_status更新任务的状态为processing
        # 2. 运行（graph.stream()）

        # 2.1 在运行图的某个节点之前调用mark_node_running
        # 2.2 在运行图的某个节点结束调用mark_node_done

        # graph.stream()执行完所有节点 update_task_status更新任务的状态为completed
        try:
            # 1. 标记任务开始处理
            self._task_service.update_task_status(task_id, "processing")

            # 2. 构建 LangGraph 初始状态
            global_graph_init_status: ImportGraphState = {
                "task_id": task_id,
                "file_dir": file_dir,
                "import_file_path": import_file_path
            }

            # 3. 流式执行整个导入流水线
            for event in kb_import_graph_app.stream(global_graph_init_status):
                for key, value in event.items():
                    print(f"[{task_id}] Completed Node: {key}")

            # 4. 标记任务完成
            self._task_service.update_task_status(task_id, "completed")

        except Exception as e:
            self._task_service.update_task_status(task_id, "failed")
            # 兜底清理：异常可能发生在节点之外(如 LangGraph 调度/并行合并),
            # 此时 BaseNode 的 except 不会执行, running_list 会残留「正在处理」记录
            self._task_service.clear_running(task_id)
            print(f"[{task_id}] Error: {e}")

