"""
Landsat 8 影像一键预处理模块 - 兼容性封装

这是一个兼容性封装，用于保持向后兼容。
实际的实现已经模块化到 src/ 目录中。

使用方法:
    from landsat8_processor import Landsat8Processor
    processor = Landsat8Processor()

新的模块化导入方式:
    from src.core.processor import Landsat8Processor
"""

# 从模块化实现导入所有功能
from src.core.processor import Landsat8Processor
from src.core.constants import (
    RADIANCE_MULT,
    RADIANCE_ADD,
    ESUN,
    SIXS_PARAMETERS,
    COMPOSITE_MAP
)

# 导出所有内容
__all__ = [
    'Landsat8Processor',
    'RADIANCE_MULT',
    'RADIANCE_ADD',
    'ESUN',
    'SIXS_PARAMETERS',
    'COMPOSITE_MAP'
]

__version__ = "1.0.2"
__doc__ = """
Landsat 8 影像一键预处理模块 (v1.0.2)

本模块提供完整的 Landsat 8 影像预处理功能，包括:
- 辐射定标 (DN → 辐射亮度 → 地表反射率)
- 大气校正 (DOS暗目标法 / 6S模型)
- 云掩膜处理
- 影像裁剪 (按范围或矢量)
- 波段合成 (真彩色/假彩色/NDVI等)
- 预览图生成

项目结构:
- src/core/        核心模块 (处理器、常量、模型)
- src/operations/  处理操作 (辐射定标、大气校正、几何处理、合成)
- src/services/    服务层 (进度管理、文件管理)
- src/api/         API层 (FastAPI路由和配置)
"""
