"""进度管理服务"""

import threading
from datetime import datetime
from typing import Dict, Optional, Union
from ..core.models import ProgressRecord, ProgressStep, ProcessingResult
from ..core.constants import PROGRESS_STEPS


class ProgressManager:
    """进度管理器"""

    def __init__(self):
        self.progress_tasks: Dict[str, ProgressRecord] = {}
        self.lock = threading.Lock()

    def _build_step_state(self) -> list:
        """构建步骤状态列表"""
        return [ProgressStep(**step) for step in PROGRESS_STEPS]

    def init_progress(self, job_id: str) -> ProgressRecord:
        """初始化进度记录"""
        record = ProgressRecord(
            job_id=job_id,
            status="pending",
            progress=0,
            current_step=None,
            detail="等待开始",
            steps=self._build_step_state(),
            created_at=datetime.utcnow().isoformat() + "Z",
            updated_at=datetime.utcnow().isoformat() + "Z",
        )
        with self.lock:
            self.progress_tasks[job_id] = record
        return record

    def update_progress(self,
                       job_id: str,
                       step_id: str = None,
                       step_status: str = None,
                       progress: int = None,
                       detail: str = None,
                       status: str = None,
                       result: Union[dict, ProcessingResult] = None,
                       error: str = None):
        """更新进度"""
        with self.lock:
            task = self.progress_tasks.get(job_id)
            if not task:
                return

            if step_id:
                if step_status == 'active':
                    for step in task.steps:
                        if step.status == 'active' and step.id != step_id:
                            step.status = 'completed'

                for step in task.steps:
                    if step.id == step_id:
                        if step_status:
                            step.status = step_status
                        if detail:
                            step.detail = detail
                        step.time = datetime.now().strftime("%H:%M:%S")
                task.current_step = step_id

            if progress is not None:
                task.progress = progress

            if detail:
                task.detail = detail

            if result is not None:
                # Convert dict to ProcessingResult if needed
                if isinstance(result, dict):
                    task.result = ProcessingResult(**result)
                else:
                    task.result = result

            if error:
                task.error = error

            if status:
                task.status = status
                if status == 'success':
                    for step in task.steps:
                        if step.status not in ['completed', 'exception']:
                            step.status = 'completed'
                    task.progress = 100
                if status == 'error':
                    for step in task.steps:
                        if step.status == 'active':
                            step.status = 'exception'
                    task.progress = max(task.progress, 100)

            task.updated_at = datetime.utcnow().isoformat() + "Z"

    def get_progress(self, job_id: str) -> Optional[ProgressRecord]:
        """获取进度记录"""
        with self.lock:
            return self.progress_tasks.get(job_id)

    def remove_progress(self, job_id: str):
        """移除进度记录"""
        with self.lock:
            self.progress_tasks.pop(job_id, None)
