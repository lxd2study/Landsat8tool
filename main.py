"""
Landsat 8 影像预处理 FastAPI 服务

提供完整的 Landsat 8 影像预处理功能，包括：
- 辐射定标 (DN → 辐射亮度 → 地表反射率)
- DOS 大气校正
- 云掩膜处理
- 影像裁剪 (按范围或矢量)
- 波段合成 (真彩色/假彩色等)
"""

import os
import sys
import logging
import tempfile
import shutil
import threading
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
from landsat8_processor import Landsat8Processor


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Landsat 8 影像预处理服务",
    description="提供 Landsat 8 影像的完整预处理流程，包括辐射定标、大气校正、云掩膜、裁剪和波段合成",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/libs", StaticFiles(directory="libs"), name="libs")

@app.get('/')
def serve_index():
    """提供前端页面"""
    index_path = os.path.join('index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {
            "service": "Landsat 8 影像预处理服务",
            "version": "1.0.0",
            "message": "前端文件未找到，请访问 /docs 查看API文档",
            "endpoints": [
                "/preprocess_landsat8",
                "/composite_types",
                "/band_info",
                "/read_mtl",
                "/extract_cloud_mask",
                "/docs",
                "/redoc"
            ]
        }

@app.get('/health')
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "landsat8-processor",
        "version": "1.0.0"
    }


@app.get('/composite_types')
def get_composite_types():
    """
    获取支持的合成影像类型列表

    返回所有可用的波段合成方案及其描述
    """
    return {
        "composite_types": [
            {
                "type": "true_color",
                "name": "真彩色",
                "bands": ["B4", "B3", "B2"],
                "description": "真实自然色彩 (Red-Green-Blue)",
                "use_case": "自然地物识别，最接近人眼视觉"
            },
            {
                "type": "false_color",
                "name": "假彩色",
                "bands": ["B5", "B4", "B3"],
                "description": "植被增强 (NIR-Red-Green)",
                "use_case": "植被健康监测，植物呈红色"
            },
            {
                "type": "agriculture",
                "name": "农业监测",
                "bands": ["B6", "B5", "B2"],
                "description": "农作物分析 (SWIR1-NIR-Blue)",
                "use_case": "农作物类型识别，土壤湿度评估"
            },
            {
                "type": "urban",
                "name": "城市研究",
                "bands": ["B7", "B6", "B4"],
                "description": "城市区域增强 (SWIR2-SWIR1-Red)",
                "use_case": "建筑物识别，城市规划"
            },
            {
                "type": "natural_color",
                "name": "自然彩色",
                "bands": ["B5", "B4", "B3"],
                "description": "自然色调 (NIR-Red-Green)",
                "use_case": "地形地貌分析"
            },
            {
                "type": "swir",
                "name": "短波红外",
                "bands": ["B7", "B5", "B3"],
                "description": "短波红外合成 (SWIR2-NIR-Green)",
                "use_case": "水体识别，云雾穿透"
            },
            {
                "type": "ndvi",
                "name": "植被指数",
                "bands": ["B5", "B4"],
                "description": "归一化植被指数 (NIR-Red)/(NIR+Red)",
                "use_case": "植被覆盖度，植被健康状况"
            }
        ]
    }


@app.get('/band_info')
def get_band_info():
    """
    获取Landsat 8波段信息

    返回所有波段的详细信息，包括波长范围和用途
    """
    return {
        "satellite": "Landsat 8",
        "sensor": "OLI/TIRS",
        "bands": [
            {
                "band": "B1",
                "name": "海岸/气溶胶",
                "wavelength": "0.43-0.45 μm",
                "resolution": "30m",
                "use": "浅水映射，气溶胶研究"
            },
            {
                "band": "B2",
                "name": "蓝",
                "wavelength": "0.45-0.51 μm",
                "resolution": "30m",
                "use": "水体穿透，土壤/植被识别"
            },
            {
                "band": "B3",
                "name": "绿",
                "wavelength": "0.53-0.59 μm",
                "resolution": "30m",
                "use": "植被活力评估"
            },
            {
                "band": "B4",
                "name": "红",
                "wavelength": "0.64-0.67 μm",
                "resolution": "30m",
                "use": "植被分类，生物量估算"
            },
            {
                "band": "B5",
                "name": "近红外",
                "wavelength": "0.85-0.88 μm",
                "resolution": "30m",
                "use": "生物量，海岸线映射"
            },
            {
                "band": "B6",
                "name": "短波红外1",
                "wavelength": "1.57-1.65 μm",
                "resolution": "30m",
                "use": "土壤湿度，植被含水量"
            },
            {
                "band": "B7",
                "name": "短波红外2",
                "wavelength": "2.11-2.29 μm",
                "resolution": "30m",
                "use": "岩石/矿物识别，云雾穿透"
            },
            {
                "band": "B8",
                "name": "全色",
                "wavelength": "0.50-0.68 μm",
                "resolution": "15m",
                "use": "高分辨率影像锐化"
            },
            {
                "band": "B9",
                "name": "卷云",
                "wavelength": "1.36-1.38 μm",
                "resolution": "30m",
                "use": "卷云检测"
            },
            {
                "band": "B10",
                "name": "热红外1",
                "wavelength": "10.6-11.19 μm",
                "resolution": "100m",
                "use": "地表温度"
            },
            {
                "band": "B11",
                "name": "热红外2",
                "wavelength": "11.5-12.51 μm",
                "resolution": "100m",
                "use": "地表温度"
            }
        ]
    }


@app.post('/read_mtl')
async def read_mtl(
    mtl_file: UploadFile = File(..., description="MTL元数据文件")
):
    """
    读取并解析MTL元数据文件

    参数:
    - mtl_file: MTL元数据文件 (*_MTL.txt)

    返回MTL文件中的关键元数据信息
    """
    logger.info(f"接收MTL文件解析请求: {mtl_file.filename}")

    # 创建临时文件
    temp_dir = tempfile.mkdtemp(prefix="mtl_")
    mtl_path = os.path.join(temp_dir, "MTL.txt")

    try:
        # 保存MTL文件
        content = await mtl_file.read()
        with open(mtl_path, 'wb') as f:
            f.write(content)

        # 解析MTL文件
        processor = Landsat8Processor()
        metadata = processor.read_mtl_file(mtl_path)

        logger.info(f"MTL文件解析成功: {metadata.get('scene_id', 'Unknown')}")

        return {
            "status": "success",
            "filename": mtl_file.filename,
            "metadata": metadata,
            "message": "MTL文件解析成功"
        }

    except Exception as e:
        logger.error(f"MTL文件解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MTL文件解析失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")


@app.post('/extract_cloud_mask')
async def extract_cloud_mask(
    qa_band: UploadFile = File(..., description="QA波段文件"),
    confidence_threshold: str = Form('medium', description="置信度阈值: low, medium, high")
):
    """
    从QA波段提取云掩膜

    参数:
    - qa_band: QA波段文件 (*_BQA.TIF)
    - confidence_threshold: 云检测置信度阈值

    返回云掩膜统计信息
    """
    logger.info(f"接收云掩膜提取请求: {qa_band.filename}")

    temp_dir = tempfile.mkdtemp(prefix="cloud_mask_")
    qa_path = os.path.join(temp_dir, qa_band.filename)

    try:
        # 保存QA文件
        content = await qa_band.read()
        with open(qa_path, 'wb') as f:
            f.write(content)

        # 提取云掩膜
        processor = Landsat8Processor()
        cloud_mask = processor.cloud_mask_from_qa(qa_path, confidence_threshold)

        # 计算统计信息
        total_pixels = cloud_mask.size
        cloud_pixels = np.sum(cloud_mask == 1)
        clear_pixels = total_pixels - cloud_pixels
        cloud_percentage = (cloud_pixels / total_pixels) * 100

        logger.info(f"云掩膜提取成功，云覆盖率: {cloud_percentage:.2f}%")

        return {
            "status": "success",
            "filename": qa_band.filename,
            "statistics": {
                "total_pixels": int(total_pixels),
                "cloud_pixels": int(cloud_pixels),
                "clear_pixels": int(clear_pixels),
                "cloud_percentage": round(cloud_percentage, 2),
                "confidence_threshold": confidence_threshold
            },
            "shape": [int(cloud_mask.shape[0]), int(cloud_mask.shape[1])],
            "message": f"云掩膜提取成功，云覆盖率: {cloud_percentage:.2f}%"
        }

    except Exception as e:
        logger.error(f"云掩膜提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"云掩膜提取失败: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")


@app.get('/select_folder')
def select_folder():
    """
    打开系统文件选择对话框，让用户选择输出文件夹

    返回:
    - path: 选择的文件夹绝对路径
    - status: 选择状态
    """
    logger.info("接收到文件夹选择请求")

    selected_path = None
    error_message = None

    def open_dialog():
        nonlocal selected_path, error_message
        try:
            # 创建隐藏的 Tkinter 窗口
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            root.attributes('-topmost', True)  # 置顶显示对话框

            # 打开文件夹选择对话框
            folder_path = filedialog.askdirectory(
                title="选择输出文件夹",
                initialdir=os.path.expanduser("~")  # 从用户主目录开始
            )

            if folder_path:
                selected_path = folder_path
                logger.info(f"用户选择了文件夹: {folder_path}")
            else:
                logger.info("用户取消了文件夹选择")

            root.destroy()
        except Exception as e:
            error_message = str(e)
            logger.error(f"文件夹选择失败: {str(e)}")

    # 在新线程中运行对话框，避免阻塞
    dialog_thread = threading.Thread(target=open_dialog)
    dialog_thread.start()
    dialog_thread.join(timeout=60)  # 最多等待60秒

    if error_message:
        raise HTTPException(status_code=500, detail=f"打开文件选择对话框失败: {error_message}")

    if selected_path:
        return {
            "status": "success",
            "path": selected_path,
            "message": "文件夹选择成功"
        }
    else:
        return {
            "status": "cancelled",
            "path": None,
            "message": "用户取消了选择"
        }


@app.post('/preprocess_landsat8')
async def preprocess_landsat8(
    bands: List[UploadFile] = File(..., description="Landsat 8波段文件列表"),
    mtl_file: Optional[UploadFile] = File(None, description="MTL元数据文件"),
    qa_band: Optional[UploadFile] = File(None, description="QA波段文件"),
    output_dir: str = Form(..., description="输出目录路径"),
    clip_extent: Optional[str] = Form(None, description="裁剪范围'xmin,ymin,xmax,ymax'"),
    clip_shapefile: Optional[List[UploadFile]] = File(None, description="裁剪矢量文件"),
    create_composites: Optional[str] = Form(None, description="合成类型，如'true_color,false_color'"),
    apply_cloud_mask: bool = Form(False, description="是否应用云掩膜"),
):
    """
    Landsat 8 影像一键预处理API

    功能包括:
    - 辐射定标 (DN -> 辐射亮度 -> 地表反射率)
    - 大气校正 (DOS暗目标法)
    - 云掩膜处理 (基于QA波段)
    - 影像裁剪 (按范围或矢量)
    - 波段合成 (真彩色/假彩色等)

    参数说明:
    - bands: Landsat 8 波段文件，文件名应包含波段编号如 LC08_B1.TIF, LC08_B2.TIF 等
    - mtl_file: MTL元数据文件 (*_MTL.txt)，用于提取辐射定标参数
    - qa_band: QA波段文件 (*_BQA.TIF)，用于云掩膜处理
    - output_dir: 输出目录路径
    - clip_extent: 裁剪范围 (可选)
    - clip_shapefile: 裁剪矢量文件 (可选，需包含.shp,.shx,.dbf,.prj)
    - create_composites: 合成影像类型，可选值: true_color, false_color, agriculture, urban, ndvi
    - apply_cloud_mask: 是否应用云掩膜

    返回:
    - status: 处理状态
    - processed_bands: 处理后的波段文件路径
    - composites: 合成影像文件路径
    - cloud_mask: 云掩膜文件路径
    - metadata: 影像元数据信息
    """
    logger.info(f"接收到Landsat 8预处理请求: {len(bands)}个波段文件, 输出目录={output_dir}")

    if not bands or len(bands) == 0:
        raise HTTPException(status_code=400, detail="必须上传至少一个波段文件")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="landsat8_")
    temp_band_dir = os.path.join(temp_dir, "bands")
    temp_shape_dir = os.path.join(temp_dir, "shapefile")
    os.makedirs(temp_band_dir, exist_ok=True)
    os.makedirs(temp_shape_dir, exist_ok=True)

    mtl_path = None
    qa_path = None
    band_paths = {}
    shapefile_path = None

    try:
        # 1. 保存波段文件
        logger.info("正在保存波段文件...")
        for band_file in bands:
            filename = band_file.filename

            # 从文件名中提取波段编号
            band_name = None
            for i in range(1, 12):
                if f'B{i}.' in filename.upper() or f'_B{i}.' in filename.upper():
                    band_name = f'B{i}'
                    break

            if not band_name:
                logger.warning(f"无法从文件名中识别波段编号: {filename}，跳过该文件")
                continue

            file_extension = os.path.splitext(filename)[-1]
            temp_band_path = os.path.join(temp_band_dir, f"{band_name}{file_extension}")

            content = await band_file.read()
            with open(temp_band_path, 'wb') as f:
                f.write(content)

            band_paths[band_name] = temp_band_path
            logger.info(f"已保存波段 {band_name}: {temp_band_path}")

        if len(band_paths) == 0:
            raise HTTPException(status_code=400, detail="未能识别任何有效的波段文件，请确保文件名包含波段编号(如B1, B2等)")

        logger.info(f"共识别 {len(band_paths)} 个波段文件")

        # 2. 保存MTL元数据文件
        if mtl_file:
            mtl_content = await mtl_file.read()
            mtl_path = os.path.join(temp_dir, "MTL.txt")
            with open(mtl_path, 'wb') as f:
                f.write(mtl_content)
            logger.info(f"已保存MTL文件: {mtl_path}")

        # 3. 保存QA波段文件
        if qa_band:
            qa_content = await qa_band.read()
            file_extension = os.path.splitext(qa_band.filename)[-1]
            qa_path = os.path.join(temp_dir, f"BQA{file_extension}")
            with open(qa_path, 'wb') as f:
                f.write(qa_content)
            logger.info(f"已保存QA波段: {qa_path}")

        # 4. 保存裁剪矢量文件
        if clip_shapefile and len(clip_shapefile) > 0:
            logger.info(f"正在保存裁剪矢量文件，共 {len(clip_shapefile)} 个文件...")
            for shape_file in clip_shapefile:
                filename = shape_file.filename
                file_extension = os.path.splitext(filename)[-1].lower()

                if file_extension not in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
                    logger.warning(f"不支持的矢量文件类型: {file_extension}，文件名: {filename}")
                    continue

                temp_shape_path = os.path.join(temp_shape_dir, filename)
                content = await shape_file.read()
                with open(temp_shape_path, 'wb') as f:
                    f.write(content)

                if file_extension == '.shp':
                    shapefile_path = temp_shape_path
                    logger.info(f"已保存Shapefile主文件: {shapefile_path}")

            if not shapefile_path:
                raise HTTPException(status_code=400, detail="裁剪矢量文件集合中必须包含.shp文件")

        # 5. 解析裁剪范围
        extent_list = None
        if clip_extent:
            try:
                extent_list = [float(x.strip()) for x in clip_extent.split(',')]
                if len(extent_list) != 4:
                    raise ValueError("裁剪范围必须包含4个值: xmin,ymin,xmax,ymax")
                logger.info(f"裁剪范围: {extent_list}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"裁剪范围格式错误: {str(e)}")

        # 6. 解析合成影像类型
        composite_list = None
        if create_composites:
            composite_list = [c.strip() for c in create_composites.split(',') if c.strip()]
            valid_composites = ['true_color', 'false_color', 'natural_color', 'agriculture', 'urban', 'swir', 'ndvi']
            for comp in composite_list:
                if comp not in valid_composites:
                    raise HTTPException(
                        status_code=400,
                        detail=f"不支持的合成类型: {comp}。有效类型: {', '.join(valid_composites)}"
                    )
            logger.info(f"要创建的合成影像类型: {composite_list}")

        # 7. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录已创建: {output_dir}")

        # 8. 执行预处理
        logger.info("开始执行Landsat 8预处理...")
        processor = Landsat8Processor()

        result = processor.one_click_preprocess(
            band_paths=band_paths,
            output_dir=output_dir,
            mtl_path=mtl_path,
            clip_extent=extent_list,
            clip_shapefile=shapefile_path,
            create_composites=composite_list,
            apply_cloud_mask=apply_cloud_mask and qa_path is not None,
            qa_band_path=qa_path
        )

        if result.get('status') == 'error':
            logger.error(f"预处理失败: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))

        logger.info("Landsat 8预处理成功完成")

        # 添加处理摘要
        result['summary'] = {
            'total_bands_processed': len(result.get('processed_bands', {})),
            'composites_created': len(result.get('composites', {})),
            'cloud_mask_applied': apply_cloud_mask and qa_path is not None,
            'clipped': clip_extent is not None or shapefile_path is not None,
            'output_directory': output_dir
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Landsat 8预处理异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"临时目录已删除: {temp_dir}")
            except Exception as e:
                logger.warning(f"删除临时目录失败: {str(e)}")


if __name__ == '__main__':
    """启动服务器"""
    logger.info("正在启动 Landsat 8 影像预处理服务...")
    logger.info("服务地址: http://localhost:5001")
    logger.info("API 文档: http://localhost:5001/docs")
    uvicorn.run(app, host="0.0.0.0", port=5001)
