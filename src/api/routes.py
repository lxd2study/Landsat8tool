"""API路由定义"""

import os
import logging
import uuid
import threading
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse

from ..core.constants import BAND_INFO, COMPOSITE_MAP
from ..core.models import ProcessingResult
from ..services.progress import ProgressManager
from ..services.file_manager import FileManager
from ..core.processor import Landsat8Processor

logger = logging.getLogger(__name__)


def setup_routes(app: FastAPI, progress_manager: ProgressManager, file_manager: FileManager):
    """设置API路由"""

    @app.get('/')
    def serve_index():
        """提供前端页面"""
        index_path = os.path.join('index.html')
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {
                "service": "Landsat 8 影像预处理服务",
                "version": "1.0.2",
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
        """健康检查"""
        return {
            "status": "healthy",
            "service": "landsat8-processor",
            "version": "1.0.2"
        }

    @app.get('/composite_types')
    def get_composite_types():
        """获取支持的合成影像类型列表"""
        composite_types = []
        for comp_type, bands in COMPOSITE_MAP.items():
            if comp_type == 'true_color':
                name = "真彩色"
                desc = "真实自然色彩 (Red-Green-Blue)"
                use = "自然地物识别，最接近人眼视觉"
            elif comp_type == 'false_color':
                name = "假彩色"
                desc = "植被增强 (NIR-Red-Green)"
                use = "植被健康监测，植物呈红色"
            elif comp_type == 'agriculture':
                name = "农业监测"
                desc = "农作物分析 (SWIR1-NIR-Blue)"
                use = "农作物类型识别，土壤湿度评估"
            elif comp_type == 'urban':
                name = "城市研究"
                desc = "城市区域增强 (SWIR2-SWIR1-Red)"
                use = "建筑物识别，城市规划"
            elif comp_type == 'natural_color':
                name = "自然彩色"
                desc = "自然色调 (NIR-Red-Green)"
                use = "地形地貌分析"
            elif comp_type == 'swir':
                name = "短波红外"
                desc = "短波红外合成 (SWIR2-NIR-Green)"
                use = "水体识别，云雾穿透"
            elif comp_type == 'ndvi':
                name = "植被指数"
                desc = "归一化植被指数 (NIR-Red)/(NIR+Red)"
                use = "植被覆盖度，植被健康状况"

            composite_types.append({
                "type": comp_type,
                "name": name,
                "bands": bands,
                "description": desc,
                "use_case": use
            })

        return {"composite_types": composite_types}

    @app.get('/band_info')
    def get_band_info():
        """获取波段信息"""
        return BAND_INFO

    @app.post('/read_mtl')
    async def read_mtl(mtl_file: UploadFile = File(..., description="MTL元数据文件")):
        """读取并解析MTL元数据文件"""
        logger.info(f"接收MTL文件解析请求: {mtl_file.filename}")

        temp_dir = file_manager.create_temp_dir(prefix="mtl_")
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
            file_manager.cleanup_temp_dir(temp_dir)

    @app.post('/extract_cloud_mask')
    async def extract_cloud_mask(
        qa_band: UploadFile = File(..., description="QA波段文件"),
        confidence_threshold: str = Form('medium', description="置信度阈值: low, medium, high")
    ):
        """从QA波段提取云掩膜"""
        logger.info(f"接收云掩膜提取请求: {qa_band.filename}")

        temp_dir = file_manager.create_temp_dir(prefix="cloud_mask_")
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
            file_manager.cleanup_temp_dir(temp_dir)

    @app.get('/select_folder')
    def select_folder():
        """打开系统文件选择对话框"""
        logger.info("接收到文件夹选择请求")

        try:
            selected_path = file_manager.select_folder_dialog()

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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"打开文件选择对话框失败: {str(e)}")

    @app.post('/preview_raster')
    async def preview_raster(
        file_path: str = Form(..., description="待预览的栅格/合成影像路径"),
        max_size: int = Form(512, description="最大预览边长像素")
    ):
        """返回指定栅格的缩略图（base64 PNG）"""
        allowed_ext = ('.tif', '.tiff', '.img', '.png')
        if not file_path.lower().endswith(allowed_ext):
            raise HTTPException(status_code=400, detail="仅支持 .tif/.tiff/.img/.png 文件")

        try:
            processor = Landsat8Processor()
            preview = processor.build_preview_base64(file_path, max_size=max_size)
            return {"status": "success", "preview": preview}
        except Exception as e:
            logger.error(f"预览栅格失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"预览失败: {str(e)}")

    @app.get('/preprocess_landsat8_status/{job_id}')
    def preprocess_landsat8_status(job_id: str):
        """查询预处理任务进度"""
        task = progress_manager.get_progress(job_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在或已超时")
        return task

    @app.post('/preprocess_landsat8_async')
    async def preprocess_landsat8_async(
        bands: List[UploadFile] = File(..., description="Landsat 8波段文件列表"),
        mtl_file: Optional[UploadFile] = File(None, description="MTL元数据文件"),
        qa_band: Optional[UploadFile] = File(None, description="QA波段文件"),
        output_dir: str = Form(..., description="输出目录路径"),
        clip_extent: Optional[str] = Form(None, description="裁剪范围'xmin,ymin,xmax,ymax'"),
        clip_shapefile: Optional[List[UploadFile]] = File(None, description="裁剪矢量文件"),
        create_composites: Optional[str] = Form(None, description="合成类型，如'true_color,false_color'"),
        apply_cloud_mask: bool = Form(False, description="是否应用云掩膜"),
    ):
        """开始异步预处理任务"""
        if not bands or len(bands) == 0:
            raise HTTPException(status_code=400, detail="必须上传至少一个波段文件")

        job_id = str(uuid.uuid4())
        progress_manager.init_progress(job_id)

        temp_dir = file_manager.create_temp_dir(prefix=f"landsat8_{job_id}_")
        temp_band_dir = os.path.join(temp_dir, "bands")
        temp_shape_dir = os.path.join(temp_dir, "shapefile")
        os.makedirs(temp_band_dir, exist_ok=True)
        os.makedirs(temp_shape_dir, exist_ok=True)

        mtl_path = None
        qa_path = None
        band_paths = {}
        shapefile_path = None

        try:
            # 保存波段文件
            logger.info("正在保存波段文件(异步)...")
            band_paths = {}
            for band_file in bands:
                filename = band_file.filename

                # 从文件名中提取波段编号
                band_name = None
                for i in range(1, 12):
                    if f'B{i}.' in filename.upper() or f'_B{i}.' in filename.upper():
                        band_name = f'B{i}'
                        break

                if not band_name:
                    logger.warning(f"无法从文件名中识别波段编号 {filename}，跳过该文件")
                    continue

                file_extension = os.path.splitext(filename)[-1]
                temp_band_path = os.path.join(temp_band_dir, f"{band_name}{file_extension}")

                content = await band_file.read()
                with open(temp_band_path, 'wb') as f:
                    f.write(content)

                band_paths[band_name] = temp_band_path

            if len(band_paths) == 0:
                raise HTTPException(status_code=400, detail="未能识别任何有效的波段文件")

            progress_manager.update_progress(
                job_id,
                status='processing',
                step_id='upload',
                step_status='completed',
                progress=10,
                detail=f"已保存 {len(band_paths)} 个波段"
            )

            # 保存MTL文件
            if mtl_file:
                mtl_content = await mtl_file.read()
                mtl_path = os.path.join(temp_dir, "MTL.txt")
                with open(mtl_path, 'wb') as f:
                    f.write(mtl_content)

            # 保存QA波段
            if qa_band:
                qa_content = await qa_band.read()
                file_extension = os.path.splitext(qa_band.filename)[-1]
                qa_path = os.path.join(temp_dir, f"BQA{file_extension}")
                with open(qa_path, 'wb') as f:
                    f.write(qa_content)

            # 保存裁剪矢量文件
            if clip_shapefile and len(clip_shapefile) > 0:
                shapefile_path = file_manager.save_shapefiles(clip_shapefile, temp_shape_dir)

            # 解析参数
            extent_list = file_manager.parse_extent(clip_extent)
            composite_list = file_manager.parse_composites(create_composites)

            os.makedirs(output_dir, exist_ok=True)

        except HTTPException:
            file_manager.cleanup_temp_dir(temp_dir)
            progress_manager.remove_progress(job_id)
            raise
        except Exception as e:
            file_manager.cleanup_temp_dir(temp_dir)
            progress_manager.remove_progress(job_id)
            logger.error(f"预处理任务初始化失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"任务初始化失败: {str(e)}")

        def run_preprocess_job():
            processor = Landsat8Processor()

            def progress_hook(payload):
                progress_manager.update_progress(
                    job_id,
                    step_id=payload.get('step'),
                    step_status=payload.get('status'),
                    progress=payload.get('progress'),
                    detail=payload.get('detail')
                )

            try:
                result = processor.one_click_preprocess(
                    band_paths=band_paths,
                    output_dir=output_dir,
                    mtl_path=mtl_path,
                    clip_extent=extent_list,
                    clip_shapefile=shapefile_path,
                    create_composites=composite_list,
                    apply_cloud_mask=apply_cloud_mask and qa_path is not None,
                    qa_band_path=qa_path,
                    progress_callback=progress_hook
                )

                if result.get('status') == 'error':
                    raise Exception(result.get('error', '未知错误'))

                result['summary'] = {
                    'total_bands_processed': len(result.get('processed_bands', {})),
                    'composites_created': len(result.get('composites', {})),
                    'cloud_mask_applied': apply_cloud_mask and qa_path is not None,
                    'clipped': extent_list is not None or shapefile_path is not None,
                    'output_directory': output_dir
                }

                progress_manager.update_progress(
                    job_id,
                    status='success',
                    step_id='finalize',
                    step_status='completed',
                    progress=100,
                    detail='处理完成',
                    result=result
                )
            except Exception as exc:
                logger.error(f"异步预处理失败: {str(exc)}", exc_info=True)
                progress_manager.update_progress(
                    job_id,
                    status='error',
                    step_id='finalize',
                    step_status='exception',
                    progress=100,
                    detail=f"处理失败: {exc}",
                    error=str(exc)
                )
            finally:
                file_manager.cleanup_temp_dir(temp_dir)

        threading.Thread(target=run_preprocess_job, daemon=True).start()

        return {"job_id": job_id, "status": "processing"}

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
        """Landsat 8 影像一键预处理API"""
        logger.info(f"接收到Landsat 8预处理请求: {len(bands)}个波段文件, 输出目录={output_dir}")

        if not bands or len(bands) == 0:
            raise HTTPException(status_code=400, detail="必须上传至少一个波段文件")

        temp_dir = file_manager.create_temp_dir(prefix="landsat8_")
        temp_band_dir = os.path.join(temp_dir, "bands")
        temp_shape_dir = os.path.join(temp_dir, "shapefile")
        os.makedirs(temp_band_dir, exist_ok=True)
        os.makedirs(temp_shape_dir, exist_ok=True)

        mtl_path = None
        qa_path = None
        band_paths = {}
        shapefile_path = None

        try:
            # 保存波段文件
            logger.info("正在保存波段文件...")
            band_paths = {}
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

            # 保存MTL文件
            if mtl_file:
                mtl_content = await mtl_file.read()
                mtl_path = os.path.join(temp_dir, "MTL.txt")
                with open(mtl_path, 'wb') as f:
                    f.write(mtl_content)
                logger.info(f"已保存MTL文件: {mtl_path}")

            # 保存QA波段
            if qa_band:
                qa_content = await qa_band.read()
                file_extension = os.path.splitext(qa_band.filename)[-1]
                qa_path = os.path.join(temp_dir, f"BQA{file_extension}")
                with open(qa_path, 'wb') as f:
                    f.write(qa_content)
                logger.info(f"已保存QA波段: {qa_path}")

            # 保存裁剪矢量文件
            if clip_shapefile and len(clip_shapefile) > 0:
                logger.info(f"正在保存裁剪矢量文件，共 {len(clip_shapefile)} 个文件...")
                shapefile_path = file_manager.save_shapefiles(clip_shapefile, temp_shape_dir)
                if shapefile_path:
                    logger.info(f"已保存Shapefile主文件: {shapefile_path}")

            # 解析参数
            extent_list = file_manager.parse_extent(clip_extent)
            composite_list = file_manager.parse_composites(create_composites)

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"输出目录已创建: {output_dir}")

            # 执行预处理
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
                'clipped': extent_list is not None or shapefile_path is not None,
                'output_directory': output_dir
            }

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Landsat 8预处理异常: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
        finally:
            file_manager.cleanup_temp_dir(temp_dir)
