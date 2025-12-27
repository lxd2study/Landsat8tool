"""Landsat 8 影像处理器核心模块"""

import os
import base64
import numpy as np
from osgeo import gdal, ogr, osr
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# 启用GDAL异常处理
gdal.UseExceptions()
warnings.filterwarnings('ignore', category=FutureWarning)

from .constants import RADIANCE_MULT, RADIANCE_ADD, ESUN, COMPOSITE_MAP
from ..operations.radiometric import dn_to_radiance, radiance_to_reflectance
from ..operations.atmospheric import dark_object_subtraction, cloud_mask_from_qa
from ..operations.geometric import clip_raster, resample_to_match
from ..operations.synthesis import create_composite, create_ndvi


class Landsat8Processor:
    """Landsat 8 影像处理器"""

    def __init__(self):
        """初始化处理器"""
        gdal.AllRegister()
        self.metadata = {}
        # 使用常量中的参数
        self.RADIANCE_MULT = RADIANCE_MULT.copy()
        self.RADIANCE_ADD = RADIANCE_ADD.copy()
        self.ESUN = ESUN.copy()

    def read_mtl_file(self, mtl_path: str) -> Dict:
        """
        读取 Landsat 8 MTL 元数据文件

        Args:
            mtl_path: MTL文件路径

        Returns:
            包含元数据的字典
        """
        metadata = {}
        try:
            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('\"')
                        metadata[key] = value

            # 提取关键参数
            self.metadata = {
                'scene_id': metadata.get('LANDSAT_SCENE_ID', ''),
                'date_acquired': metadata.get('DATE_ACQUIRED', ''),
                'sun_elevation': float(metadata.get('SUN_ELEVATION', 45.0)),
                'sun_azimuth': float(metadata.get('SUN_AZIMUTH', 135.0)),
                'cloud_cover': float(metadata.get('CLOUD_COVER', 0.0)),
            }

            # 提取各波段的辐射定标参数
            bands_found = 0
            for i in range(1, 12):
                band = f'B{i}'
                mult_key = f'RADIANCE_MULT_BAND_{i}'
                add_key = f'RADIANCE_ADD_BAND_{i}'

                if mult_key in metadata:
                    self.RADIANCE_MULT[band] = float(metadata[mult_key])
                    bands_found += 1
                if add_key in metadata:
                    self.RADIANCE_ADD[band] = float(metadata[add_key])

            print(f"从MTL文件中读取了 {bands_found} 个波段的辐射定标参数")
            print(f"B4参数: MULT={self.RADIANCE_MULT.get('B4')}, ADD={self.RADIANCE_ADD.get('B4')}")
            print(f"B3参数: MULT={self.RADIANCE_MULT.get('B3')}, ADD={self.RADIANCE_ADD.get('B3')}")
            print(f"B2参数: MULT={self.RADIANCE_MULT.get('B2')}, ADD={self.RADIANCE_ADD.get('B2')}")

            return self.metadata

        except Exception as e:
            raise Exception(f"读取MTL文件失败: {str(e)}")

    def dn_to_radiance(self, dn_array: np.ndarray, band_name: str) -> np.ndarray:
        """
        DN值转辐射亮度

        使用operations.radiometric模块的函数
        """
        return dn_to_radiance(dn_array, band_name, self.RADIANCE_MULT, self.RADIANCE_ADD)

    def radiance_to_reflectance(self, radiance: np.ndarray, band_name: str,
                               sun_elevation: float = None) -> np.ndarray:
        """
        辐射亮度转地表反射率 (TOA反射率)

        使用operations.radiometric模块的函数
        """
        if sun_elevation is None:
            sun_elevation = self.metadata.get('sun_elevation', 45.0)

        return radiance_to_reflectance(radiance, band_name, self.ESUN, sun_elevation)

    def dark_object_subtraction(self, reflectance: np.ndarray,
                               percentile: float = 1.0) -> np.ndarray:
        """
        暗目标法大气校正 (DOS)

        使用operations.atmospheric模块的函数
        """
        return dark_object_subtraction(reflectance, percentile)

    def cloud_mask_from_qa(self, qa_band_path: str,
                          confidence_threshold: str = 'medium') -> np.ndarray:
        """
        从QA波段提取云掩膜

        使用operations.atmospheric模块的函数
        """
        return cloud_mask_from_qa(qa_band_path, confidence_threshold)

    def clip_raster(self, input_path: str, output_path: str,
                   extent: List[float] = None,
                   shapefile: str = None) -> str:
        """
        裁剪栅格影像

        使用operations.geometric模块的函数
        """
        return clip_raster(input_path, output_path, extent, shapefile)

    def pansharpening(self, pan_path: str, multi_paths: List[str],
                     output_path: str, method: str = 'brovey') -> str:
        """
        全色与多光谱融合

        使用operations.geometric模块的函数
        """
        from ..operations.geometric import pansharpening
        return pansharpening(pan_path, multi_paths, output_path, method)

    def process_band(self, band_path: str, band_name: str,
                    apply_atm_correction: bool = True) -> np.ndarray:
        """
        处理单个波段: DN -> 辐射亮度 -> 反射率 -> 大气校正

        Args:
            band_path: 波段文件路径
            band_name: 波段名称
            apply_atm_correction: 是否应用大气校正

        Returns:
            处理后的反射率数组
        """
        # 读取波段
        dataset = gdal.Open(band_path)
        if dataset is None:
            raise Exception(f"无法打开波段文件: {band_path}")

        dn = dataset.ReadAsArray()

        # 检查数组是否为空
        if dn.size == 0:
            raise Exception(f"波段文件为空: {band_path}")

        print(f"波段 {band_name} DN统计: min={dn.min()}, max={dn.max()}, mean={dn.mean():.2f}, 形状={dn.shape}")

        # DN -> 辐射亮度
        radiance = self.dn_to_radiance(dn, band_name)
        print(f"波段 {band_name} 辐射亮度统计: min={radiance.min():.6f}, max={radiance.max():.6f}, mean={radiance.mean():.6f}")

        # 辐射亮度 -> 反射率
        reflectance = self.radiance_to_reflectance(radiance, band_name)
        print(f"波段 {band_name} 反射率统计: min={reflectance.min():.6f}, max={reflectance.max():.6f}, mean={reflectance.mean():.6f}")

        # 大气校正
        if apply_atm_correction:
            reflectance = self.dark_object_subtraction(reflectance)
            print(f"波段 {band_name} 大气校正后统计: min={reflectance.min():.6f}, max={reflectance.max():.6f}, mean={reflectance.mean():.6f}")

        dataset = None
        return reflectance

    def create_composite(self, band_paths: Dict[str, str],
                        output_path: str,
                        composite_type: str = 'true_color',
                        scale_factor: float = 255.0) -> str:
        """
        创建波段合成影像

        使用operations.synthesis模块的函数
        """
        return create_composite(band_paths, output_path, composite_type, scale_factor)

    def create_ndvi(self, band_paths: Dict[str, str], output_path: str) -> str:
        """
        创建NDVI (归一化植被指数) 影像

        使用operations.synthesis模块的函数
        """
        return create_ndvi(band_paths, output_path)

    def one_click_preprocess(self,
                            band_paths: Dict[str, str],
                            output_dir: str,
                            mtl_path: str = None,
                            clip_extent: List[float] = None,
                            clip_shapefile: str = None,
                            create_composites: List[str] = None,
                            apply_cloud_mask: bool = False,
                            qa_band_path: str = None,
                            progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict:
        """
        一键预处理主函数
        """
        def report(step_id: str, detail: str, progress: Optional[int] = None, status: str = 'active'):
            if progress_callback:
                progress_callback({
                    'step': step_id,
                    'detail': detail,
                    'progress': progress,
                    'status': status
                })

        results = {
            'status': 'success',
            'processed_bands': {},
            'composites': {},
            'cloud_mask': None,
            'metadata': {}
        }

        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            report('prepare_output', '•', progress=15, status='completed')

            # 读取元数据
            if mtl_path and os.path.exists(mtl_path):
                print(f"正在读取MTL文件: {mtl_path}")
                self.read_mtl_file(mtl_path)
                results['metadata'] = self.metadata
                print(f"MTL元数据读取完成，太阳高度角: {self.metadata.get('sun_elevation', 'N/A')}")
                print(f"辐射定标参数示例 - B4增益: {self.RADIANCE_MULT.get('B4', 'N/A')}, B4偏移: {self.RADIANCE_ADD.get('B4', 'N/A')}")
                report('metadata', '已读取MTL元数据', progress=25, status='completed')
            else:
                print("警告: 未找到MTL文件，使用默认参数")
                report('metadata', '未提供MTL，使用默认参数', progress=20, status='completed')

            # 处理云掩膜
            cloud_mask = None
            qa_band_ref_path = None
            if apply_cloud_mask and qa_band_path and os.path.exists(qa_band_path):
                try:
                    report('cloud_mask', '正在提取云掩膜', progress=35, status='active')
                    cloud_mask = self.cloud_mask_from_qa(qa_band_path)
                    qa_band_ref_path = qa_band_path
                    cloud_mask_path = os.path.join(output_dir, 'cloud_mask.tif')

                    # 保存云掩膜
                    ref_ds = gdal.Open(list(band_paths.values())[0])
                    driver = gdal.GetDriverByName('GTiff')
                    mask_ds = driver.Create(cloud_mask_path,
                                           ref_ds.RasterXSize,
                                           ref_ds.RasterYSize,
                                           1,
                                           gdal.GDT_Byte)
                    mask_ds.SetProjection(ref_ds.GetProjection())
                    mask_ds.SetGeoTransform(ref_ds.GetGeoTransform())

                    # 如果云掩膜尺寸不匹配，进行重采样
                    if cloud_mask.shape != (ref_ds.RasterYSize, ref_ds.RasterXSize):
                        original_mask_ds = gdal.Open(qa_band_path)
                        gdal.Warp(mask_ds, original_mask_ds, resampleAlg=gdal.GRA_NearestNeighbour)
                        cloud_mask = mask_ds.ReadAsArray()
                        original_mask_ds = None
                    else:
                        mask_ds.GetRasterBand(1).WriteArray(cloud_mask)

                    mask_ds = None
                    ref_ds = None

                    results['cloud_mask'] = cloud_mask_path
                    report('cloud_mask', '云掩膜处理完成', progress=45, status='completed')
                except Exception as e:
                    print(f"云掩膜处理失败，跳过云掩膜: {str(e)}")
                    cloud_mask = None
                    qa_band_ref_path = None
                    report('cloud_mask', '云掩膜处理失败，已跳过', progress=45, status='exception')
            else:
                report('cloud_mask', '未启用云掩膜或未提供QA文件', progress=35, status='completed')

            # 处理每个波段
            total_bands = max(len(band_paths), 1)
            processed_count = 0
            report('bands', '开始处理波段', progress=50, status='active')
            for band_name, band_path in band_paths.items():
                if not os.path.exists(band_path):
                    continue

                # 辐射定标和大气校正
                reflectance = self.process_band(band_path, band_name, apply_atm_correction=True)

                # 应用云掩膜
                if cloud_mask is not None and cloud_mask.shape == reflectance.shape:
                    reflectance[cloud_mask == 1] = 0
                elif cloud_mask is not None:
                    print(f"波段 {band_name} 与云掩膜尺寸不匹配，跳过云掩膜处理")

                # 保存处理后的波段
                output_band_path = os.path.join(output_dir, f'{band_name}_processed.tif')

                ref_ds = gdal.Open(band_path)
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(output_band_path,
                                      ref_ds.RasterXSize,
                                      ref_ds.RasterYSize,
                                      1,
                                      gdal.GDT_Float32,
                                      options=['COMPRESS=LZW'])

                out_ds.SetProjection(ref_ds.GetProjection())
                out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
                out_ds.GetRasterBand(1).WriteArray(reflectance)
                out_ds.GetRasterBand(1).SetNoDataValue(0)

                ref_ds = None
                out_ds = None

                # 裁剪
                if clip_extent or clip_shapefile:
                    clipped_path = os.path.join(output_dir, f'{band_name}_clipped.tif')
                    self.clip_raster(output_band_path, clipped_path,
                                   extent=clip_extent,
                                   shapefile=clip_shapefile)
                    results['processed_bands'][band_name] = clipped_path
                else:
                    results['processed_bands'][band_name] = output_band_path

                processed_count += 1
                band_progress = 50 + int(30 * processed_count / total_bands)
                report('bands', f"已处理波段 {band_name}", progress=band_progress, status='active')

            if clip_extent or clip_shapefile:
                report('clip', '影像裁剪完成', progress=82, status='completed')
            else:
                report('clip', '无裁剪任务，跳过', progress=80, status='completed')

            # 创建合成影像
            if create_composites:
                report('composite', '正在生成合成影像', progress=88, status='active')
                for composite_type in create_composites:
                    composite_path = os.path.join(output_dir, f'{composite_type}.tif')
                    self.create_composite(results['processed_bands'],
                                        composite_path,
                                        composite_type=composite_type)
                    results['composites'][composite_type] = composite_path
                report('composite', '合成影像生成完成', progress=96, status='completed')
            else:
                report('composite', '未选择合成影像，跳过', progress=90, status='completed')

            report('finalize', '处理完成，结果已写入输出目录', progress=100, status='completed')
            return results

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            report('finalize', f"处理失败: {str(e)}", progress=100, status='exception')
            return results

    def build_preview_base64(self, raster_path: str, max_size: int = 512) -> Dict:
        """
        生成栅格的预览图（base64 PNG）

        Args:
            raster_path: 栅格文件路径
            max_size: 最大预览尺寸（像素）

        Returns:
            包含base64编码PNG、宽度、高度和波段数的字典
        """
        if not os.path.exists(raster_path):
            raise Exception(f"文件不存在: {raster_path}")

        ds = gdal.Open(raster_path)
        if ds is None:
            raise Exception(f"无法打开文件: {raster_path}")

        width, height, bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
        source_ds = ds

        # 限制预览尺寸，避免前端过大
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            target_x = max(1, int(width * scale))
            target_y = max(1, int(height * scale))
            source_ds = gdal.Translate(
                '',
                ds,
                format='MEM',
                width=target_x,
                height=target_y,
                resampleAlg=gdal.GRA_Bilinear
            )

        arr = source_ds.ReadAsArray()
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        band_count = min(3, arr.shape[0])
        scaled_bands = []
        for i in range(band_count):
            band = arr[i]
            valid = band[np.isfinite(band)]
            if valid.size == 0:
                scaled = np.zeros_like(band, dtype=np.uint8)
            else:
                p2 = np.percentile(valid, 2)
                p98 = np.percentile(valid, 98)
                if p98 > p2:
                    norm = (band - p2) / (p98 - p2)
                else:
                    norm = band - p2
                norm = np.clip(norm, 0, 1)
                scaled = (norm * 255).astype(np.uint8)
            scaled_bands.append(scaled)

        if band_count == 1:
            scaled_bands = [scaled_bands[0], scaled_bands[0], scaled_bands[0]]
            band_count = 3

        mem_drv = gdal.GetDriverByName('MEM')
        out_ds = mem_drv.Create(
            '',
            scaled_bands[0].shape[1],
            scaled_bands[0].shape[0],
            band_count,
            gdal.GDT_Byte
        )

        for idx in range(band_count):
            out_ds.GetRasterBand(idx + 1).WriteArray(scaled_bands[idx])

        png_path = '/vsimem/preview.png'
        gdal.Translate(png_path, out_ds, format='PNG')

        f = gdal.VSIFOpenL(png_path, 'rb')
        gdal.VSIFSeekL(f, 0, 2)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        data = gdal.VSIFReadL(1, size, f)
        gdal.VSIFCloseL(f)
        gdal.Unlink(png_path)

        if source_ds is not ds:
            source_ds = None
        ds = None
        out_ds = None

        return {
            'base64': base64.b64encode(data).decode('utf-8'),
            'width': int(width),
            'height': int(height),
            'bands': int(bands)
        }
