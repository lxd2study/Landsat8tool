"""数据模型和类型定义"""

from typing import Dict, List, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel


class CompositeType(str, Enum):
    """合成影像类型"""
    TRUE_COLOR = "true_color"
    FALSE_COLOR = "false_color"
    AGRICULTURE = "agriculture"
    URBAN = "urban"
    NATURAL_COLOR = "natural_color"
    SWIR = "swir"
    NDVI = "ndvi"


class BandPaths(BaseModel):
    """波段路径模型"""
    bands: Dict[str, str]


class ProcessingResult(BaseModel):
    """处理结果模型"""
    status: str = "success"
    processed_bands: Dict[str, str] = {}
    composites: Dict[str, str] = {}
    cloud_mask: Optional[str] = None
    metadata: Dict[str, Union[str, float]] = {}
    summary: Optional[Dict] = None
    error: Optional[str] = None


class ProgressStep(BaseModel):
    """进度步骤模型"""
    id: str
    title: str
    detail: str
    status: str = "pending"
    time: str = ""


class ProgressRecord(BaseModel):
    """进度记录模型"""
    job_id: str
    status: str
    progress: int
    current_step: Optional[str]
    detail: str
    steps: List[ProgressStep]
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class Metadata(BaseModel):
    """元数据模型"""
    scene_id: str = ""
    date_acquired: str = ""
    sun_elevation: float = 45.0
    sun_azimuth: float = 135.0
    cloud_cover: float = 0.0
