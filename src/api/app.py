"""FastAPI应用配置"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..core.constants import PROGRESS_STEPS
from ..services.progress import ProgressManager
from ..services.file_manager import FileManager
from .routes import setup_routes


def create_app() -> FastAPI:
    """创建并配置FastAPI应用"""

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # 创建应用
    app = FastAPI(
        title="Landsat 8 影像预处理服务",
        description="提供 Landsat 8 影像的完整预处理流程，包括辐射定标、大气校正、云掩膜、裁剪和波段合成",
        version="1.0.2",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件
    app.mount("/libs", StaticFiles(directory="libs"), name="libs")

    # 初始化服务
    progress_manager = ProgressManager()
    file_manager = FileManager()

    # 设置路由
    setup_routes(app, progress_manager, file_manager)

    return app
