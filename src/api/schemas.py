"""API数据模型"""

from typing import List, Optional, Dict
from pydantic import BaseModel


class MTLReadResponse(BaseModel):
    """MTL文件读取响应"""
    status: str
    filename: str
    metadata: Dict
    message: str


class CloudMaskResponse(BaseModel):
    """云掩膜提取响应"""
    status: str
    filename: str
    statistics: Dict
    shape: List[int]
    message: str


class FolderSelectResponse(BaseModel):
    """文件夹选择响应"""
    status: str
    path: Optional[str]
    message: str


class PreprocessResponse(BaseModel):
    """预处理响应"""
    status: str
    processed_bands: Dict[str, str]
    composites: Dict[str, str]
    cloud_mask: Optional[str]
    metadata: Dict
    summary: Optional[Dict]


class AsyncJobResponse(BaseModel):
    """异步任务响应"""
    job_id: str
    status: str


class ProgressResponse(BaseModel):
    """进度查询响应"""
    job_id: str
    status: str
    progress: int
    current_step: Optional[str]
    detail: str
    steps: List[Dict]
    result: Optional[Dict]
    error: Optional[str]
    created_at: str
    updated_at: str


class CompositeTypeResponse(BaseModel):
    """合成类型响应"""
    composite_types: List[Dict]


class BandInfoResponse(BaseModel):
    """波段信息响应"""
    satellite: str
    sensor: str
    bands: List[Dict]
