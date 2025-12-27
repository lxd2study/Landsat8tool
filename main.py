import sys
import os

# 确保src目录在Python路径中
sys.path.insert(0, os.path.dirname(__file__))

try:
    import uvicorn
    import logging

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    from src.api.app import create_app

    # 创建应用
    app = create_app()

    if __name__ == '__main__':
        logger.info("=" * 60)
        logger.info("Landsat 8 影像预处理服务 (模块化版本)")
        logger.info("=" * 60)
        logger.info("服务地址: http://localhost:5001")
        logger.info("API文档: http://localhost:5001/docs")
        logger.info("健康检查: http://localhost:5001/health")
        logger.info("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=5001)

except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请确保已安装所需依赖:")
    print("  pip install fastapi uvicorn numpy gdal pydantic")
    sys.exit(1)
except Exception as e:
    print(f"启动错误: {e}")
    sys.exit(1)
