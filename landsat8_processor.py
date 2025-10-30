"""
Landsat 8 影像一键预处理模块

"""

import os
import numpy as np
from osgeo import gdal, ogr, osr
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import warnings

# 启用GDAL异常处理
gdal.UseExceptions()
warnings.filterwarnings('ignore', category=FutureWarning)


class Landsat8Processor:
    """Landsat 8 影像处理器"""

    # Landsat 8 OLI 辐射定标参数 (更合理的默认值)
    RADIANCE_MULT = {
        'B1': 1.234e-02, 'B2': 1.088e-02, 'B3': 9.790e-03, 'B4': 8.612e-03,
        'B5': 1.422e-02, 'B6': 5.757e-03, 'B7': 2.041e-03, 'B8': 1.329e-02,
        'B9': 4.299e-04, 'B10': 3.342e-03, 'B11': 0.00001
    }

    RADIANCE_ADD = {
        'B1': -61.822, 'B2': -60.099, 'B3': -47.739, 'B4': -32.062,
        'B5': -31.764, 'B6': -18.886, 'B7': -8.834, 'B8': -28.439,
        'B9': -1.205, 'B10': 0.1, 'B11': 0.1
    }

    # 太阳辐照度 (W/m²/μm)
    ESUN = {
        'B1': 1895.33, 'B2': 2004.57, 'B3': 1820.75, 'B4': 1549.49,
        'B5': 951.76, 'B6': 247.55, 'B7': 85.46, 'B8': 1723.88, 'B9': 366.97
    }

    def __init__(self):
        """初始化处理器"""
        gdal.AllRegister()
        self.metadata = {}

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
                        value = value.strip().strip('"')
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

        公式: L = ML * DN + AL
        其中: L-辐射亮度, ML-增益, AL-偏移, DN-像元值

        Args:
            dn_array: DN值数组
            band_name: 波段名称 (如 'B1', 'B2')

        Returns:
            辐射亮度数组
        """
        ml = self.RADIANCE_MULT.get(band_name, 0.00001)
        al = self.RADIANCE_ADD.get(band_name, 0.1)

        radiance = ml * dn_array.astype(np.float32) + al
        
        # 确保辐射亮度为正值
        radiance = np.maximum(radiance, 0.0)
        
        print(f"DN转辐射亮度参数: ML={ml}, AL={al}")
        
        return radiance

    def radiance_to_reflectance(self, radiance: np.ndarray, band_name: str,
                               sun_elevation: float = None) -> np.ndarray:
        """
        辐射亮度转地表反射率 (TOA反射率)

        公式: ρ = (π * L * d²) / (ESUN * sin(θ))
        其中: ρ-反射率, L-辐射亮度, d-日地距离, ESUN-太阳辐照度, θ-太阳高度角

        Args:
            radiance: 辐射亮度数组
            band_name: 波段名称
            sun_elevation: 太阳高度角(度)

        Returns:
            反射率数组 (0-1范围)
        """
        if sun_elevation is None:
            sun_elevation = self.metadata.get('sun_elevation', 45.0)

        # 日地距离校正因子 (简化处理,实际应根据采集日期计算)
        d = 1.0

        # 太阳辐照度
        esun = self.ESUN.get(band_name, 1500.0)

        # 太阳高度角转弧度
        sun_elevation_rad = np.deg2rad(sun_elevation)

        # 计算反射率
        reflectance = (np.pi * radiance * d * d) / (esun * np.sin(sun_elevation_rad))

        # 限制在合理范围，但允许小的负值用于后续处理
        reflectance = np.clip(reflectance, -0.1, 2.0)
        
        print(f"反射率计算参数: ESUN={esun}, 太阳高度角={sun_elevation}°")

        return reflectance

    def dark_object_subtraction(self, reflectance: np.ndarray,
                               percentile: float = 1.0) -> np.ndarray:
        """
        暗目标法大气校正 (DOS)

        Args:
            reflectance: 反射率数组
            percentile: 用于确定暗目标的百分位数

        Returns:
            大气校正后的反射率
        """
        # 获取大于0的反射率值
        positive_reflectance = reflectance[reflectance > 0]
        
        # 检查是否有有效值
        if len(positive_reflectance) == 0:
            # 如果没有有效值，直接返回原数组
            return reflectance
        
        # 计算暗目标值
        dark_value = np.percentile(positive_reflectance, percentile)

        # 减去暗目标值
        corrected = reflectance - dark_value

        # 归一化到0-1
        corrected = np.clip(corrected, 0, 1)

        return corrected

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

    def cloud_mask_from_qa(self, qa_band_path: str,
                          confidence_threshold: str = 'medium') -> np.ndarray:
        """
        从QA波段提取云掩膜

        Args:
            qa_band_path: QA波段路径
            confidence_threshold: 置信度阈值 ('low', 'medium', 'high')

        Returns:
            云掩膜数组 (0-无云, 1-有云)
        """
        dataset = gdal.Open(qa_band_path)
        if dataset is None:
            raise Exception(f"无法打开QA波段: {qa_band_path}")

        qa = dataset.ReadAsArray()
        dataset = None
        
        # 检查数组是否为空
        if qa.size == 0:
            raise Exception(f"QA波段文件为空: {qa_band_path}")

        # Landsat 8 QA位定义
        # Bit 4: Cloud (1=Yes, 0=No)
        # Bits 5-6: Cloud Confidence (00=Not Determined, 01=Low, 10=Medium, 11=High)

        cloud_bit = 4
        cloud_confidence_bits = [5, 6]

        # 提取云标记
        cloud = np.bitwise_and(qa, 1 << cloud_bit) > 0

        # 提取云置信度
        conf_value = (np.bitwise_and(qa, 1 << cloud_confidence_bits[0]) > 0).astype(int) + \
                     (np.bitwise_and(qa, 1 << cloud_confidence_bits[1]) > 0).astype(int) * 2

        # 根据置信度阈值创建掩膜
        threshold_map = {'low': 1, 'medium': 2, 'high': 3}
        threshold = threshold_map.get(confidence_threshold, 2)

        cloud_mask = np.logical_or(cloud, conf_value >= threshold).astype(np.uint8)

        return cloud_mask

    def clip_raster(self, input_path: str, output_path: str,
                   extent: List[float] = None,
                   shapefile: str = None) -> str:
        """
        裁剪栅格影像

        Args:
            input_path: 输入影像路径
            output_path: 输出影像路径
            extent: 裁剪范围 [xmin, ymin, xmax, ymax]
            shapefile: 矢量裁剪文件

        Returns:
            输出文件路径
        """
        if shapefile:
            # 使用矢量裁剪
            gdal.Warp(output_path, input_path,
                     cutlineDSName=shapefile,
                     cropToCutline=True,
                     dstNodata=0)
        elif extent:
            # 使用范围裁剪
            gdal.Warp(output_path, input_path,
                     outputBounds=extent,
                     dstNodata=0)
        else:
            raise Exception("必须提供extent或shapefile参数")

        return output_path

    def pansharpening(self, pan_path: str, multi_paths: List[str],
                     output_path: str, method: str = 'brovey') -> str:
        """
        全色与多光谱融合

        Args:
            pan_path: 全色波段路径 (15m)
            multi_paths: 多光谱波段路径列表 (30m)
            output_path: 输出路径
            method: 融合方法 ('brovey', 'simple')

        Returns:
            输出文件路径
        """
        # 读取全色波段
        pan_ds = gdal.Open(pan_path)
        pan = pan_ds.ReadAsArray().astype(np.float32)

        # 读取多光谱波段
        multi_bands = []
        for path in multi_paths:
            ds = gdal.Open(path)
            # 重采样到全色波段分辨率
            resampled = gdal.Warp('', path,
                                 format='MEM',
                                 width=pan_ds.RasterXSize,
                                 height=pan_ds.RasterYSize,
                                 resampleAlg=gdal.GRA_Bilinear)
            multi_bands.append(resampled.ReadAsArray().astype(np.float32))

        # Brovey融合
        if method == 'brovey':
            intensity = np.mean(multi_bands, axis=0)
            intensity[intensity == 0] = 0.0001  # 避免除零

            fused_bands = []
            for band in multi_bands:
                fused = band * pan / intensity
                fused_bands.append(fused)
        else:
            # 简单平均融合
            fused_bands = []
            for band in multi_bands:
                fused = (band + pan) / 2.0
                fused_bands.append(fused)

        # 保存融合结果
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path,
                              pan_ds.RasterXSize,
                              pan_ds.RasterYSize,
                              len(fused_bands),
                              gdal.GDT_Float32)

        out_ds.SetProjection(pan_ds.GetProjection())
        out_ds.SetGeoTransform(pan_ds.GetGeoTransform())

        for i, band_data in enumerate(fused_bands, 1):
            out_band = out_ds.GetRasterBand(i)
            out_band.WriteArray(band_data)
            out_band.FlushCache()

        pan_ds = None
        out_ds = None

        return output_path

    def create_composite(self, band_paths: Dict[str, str],
                        output_path: str,
                        composite_type: str = 'true_color',
                        scale_factor: float = 255.0) -> str:
        """
        创建波段合成影像

        Args:
            band_paths: 波段路径字典 {'B1': path, 'B2': path, ...}
            output_path: 输出路径
            composite_type: 合成类型 ('true_color', 'false_color', 'ndvi')
            scale_factor: 缩放因子(转换为8bit)

        Returns:
            输出文件路径
        """
        # Landsat 8 波段映射 (修正后的正确映射)
        composite_map = {
            'true_color': ['B4', 'B3', 'B2'],      # 真彩色: Red-Green-Blue
            'false_color': ['B5', 'B4', 'B3'],     # 假彩色: NIR-Red-Green
            'agriculture': ['B6', 'B5', 'B2'],     # 农业: SWIR1-NIR-Blue
            'urban': ['B7', 'B6', 'B4'],           # 城市: SWIR2-SWIR1-Red
            'natural_color': ['B5', 'B4', 'B3'],   # 自然彩色: NIR-Red-Green
            'swir': ['B7', 'B5', 'B3'],            # 短波红外: SWIR2-NIR-Green
            'ndvi': ['B5', 'B4'],                   # NDVI: NIR-Red (特殊处理)
        }

        if composite_type not in composite_map:
            raise Exception(f"不支持的合成类型: {composite_type}")

        bands_to_use = composite_map[composite_type]

        # 特殊处理NDVI
        if composite_type == 'ndvi':
            return self.create_ndvi(band_paths, output_path)

        # 读取并处理波段
        processed_bands = []
        reference_ds = None
        band_shapes = []

        for band_name in bands_to_use:
            if band_name not in band_paths:
                raise Exception(f"缺少波段: {band_name}")

            band_path = band_paths[band_name]

            # 检查是否是已处理的文件
            if '_processed' in band_path or '_clipped' in band_path:
                # 如果是已处理的文件，直接读取作为反射率数据
                print(f"读取已处理波段 {band_name} 从: {band_path}")
                dataset = gdal.Open(band_path)
                if dataset is None:
                    raise Exception(f"无法打开已处理波段文件: {band_path}")
                
                reflectance = dataset.ReadAsArray()
                dataset = None
                
                # 确保数据在合理范围内
                reflectance = np.clip(reflectance, 0, 1)
                print(f"已处理波段 {band_name} 统计: min={reflectance.min():.6f}, max={reflectance.max():.6f}, mean={reflectance.mean():.6f}")
            else:
                # 如果是原始文件，进行完整处理
                print(f"处理原始波段 {band_name} 从: {band_path}")
                reflectance = self.process_band(band_path, band_name, apply_atm_correction=True)
            
            # 检查数组形状
            band_shapes.append(reflectance.shape)
            
            # 统计信息用于更好的拉伸
            print(f"波段 {band_name} 统计: min={reflectance.min():.6f}, max={reflectance.max():.6f}, mean={reflectance.mean():.6f}")

            # 改进的拉伸方法：使用2%线性拉伸
            valid_data = reflectance[reflectance > 0]
            if len(valid_data) > 0:
                p2 = np.percentile(valid_data, 2)
                p98 = np.percentile(valid_data, 98)
                if p98 > p2:
                    # 线性拉伸到0-1
                    stretched = np.clip((reflectance - p2) / (p98 - p2), 0, 1)
                else:
                    # 如果数据范围太小，使用原始数据
                    stretched = np.clip(reflectance, 0, 1)
            else:
                stretched = np.clip(reflectance, 0, 1)

            # 转换为8bit (0-255)
            scaled = (stretched * scale_factor).astype(np.uint8)
            processed_bands.append(scaled)

            # 保存参考数据集信息
            if reference_ds is None:
                reference_ds = gdal.Open(band_path)

        # 检查所有波段形状是否一致
        if len(set(band_shapes)) > 1:
            print(f"警告: 波段形状不一致: {band_shapes}")
            # 如果形状不一致，需要重采样到第一个波段的形状
            target_shape = band_shapes[0]
            target_width, target_height = target_shape[1], target_shape[0]
            
            for i in range(len(processed_bands)):
                if processed_bands[i].shape != target_shape:
                    # 使用GDAL进行重采样
                    import tempfile
                    temp_input = tempfile.mktemp(suffix='.tif')
                    temp_output = tempfile.mktemp(suffix='.tif')
                    
                    # 保存当前波段到临时文件
                    driver = gdal.GetDriverByName('GTiff')
                    temp_ds = driver.Create(temp_input, processed_bands[i].shape[1], processed_bands[i].shape[0], 1, gdal.GDT_Byte)
                    temp_ds.GetRasterBand(1).WriteArray(processed_bands[i])
                    temp_ds = None
                    
                    # 重采样到目标尺寸
                    gdal.Warp(temp_output, temp_input, xSize=target_width, ySize=target_height, resampleAlg=gdal.GRA_Bilinear)
                    
                    # 读取重采样后的数据
                    resampled_ds = gdal.Open(temp_output)
                    processed_bands[i] = resampled_ds.ReadAsArray()
                    resampled_ds = None
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_input)
                        os.remove(temp_output)
                    except:
                        pass

        # 验证数据
        if len(processed_bands) != 3:
            raise Exception(f"需要3个波段来创建RGB合成，当前只有{len(processed_bands)}个波段")
        
        for i, band_data in enumerate(processed_bands):
            if band_data.size == 0:
                raise Exception(f"波段{i+1}为空")
            print(f"波段{i+1}形状: {band_data.shape}, 数据类型: {band_data.dtype}, 范围: {band_data.min()}-{band_data.max()}")

        # 验证参考数据集
        if reference_ds is None:
            raise Exception("参考数据集为空")
        
        print(f"输出影像尺寸: {reference_ds.RasterXSize} x {reference_ds.RasterYSize}")

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path,
                              reference_ds.RasterXSize,
                              reference_ds.RasterYSize,
                              3,
                              gdal.GDT_Byte,
                              options=['COMPRESS=LZW', 'PHOTOMETRIC=RGB'])

        if out_ds is None:
            raise Exception(f"无法创建输出文件: {output_path}")

        out_ds.SetProjection(reference_ds.GetProjection())
        out_ds.SetGeoTransform(reference_ds.GetGeoTransform())

        # 写入波段 (注意：RGB顺序对应Red=1, Green=2, Blue=3)
        color_interpretations = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand]
        
        for i, band_data in enumerate(processed_bands):
            out_band = out_ds.GetRasterBand(i + 1)
            if out_band is None:
                raise Exception(f"无法创建波段{i+1}")
            
            # 确保数据形状匹配
            if band_data.shape != (reference_ds.RasterYSize, reference_ds.RasterXSize):
                print(f"警告: 波段{i+1}形状{band_data.shape}与输出尺寸不匹配{(reference_ds.RasterYSize, reference_ds.RasterXSize)}")
                # 使用GDAL调整数据形状
                import tempfile
                temp_input = tempfile.mktemp(suffix='.tif')
                temp_output = tempfile.mktemp(suffix='.tif')
                
                # 保存当前波段到临时文件
                temp_driver = gdal.GetDriverByName('GTiff')
                temp_ds = temp_driver.Create(temp_input, band_data.shape[1], band_data.shape[0], 1, gdal.GDT_Byte)
                temp_ds.GetRasterBand(1).WriteArray(band_data)
                temp_ds = None
                
                # 重采样到目标尺寸
                gdal.Warp(temp_output, temp_input, 
                         xSize=reference_ds.RasterXSize, 
                         ySize=reference_ds.RasterYSize, 
                         resampleAlg=gdal.GRA_Bilinear)
                
                # 读取重采样后的数据
                resampled_ds = gdal.Open(temp_output)
                band_data = resampled_ds.ReadAsArray()
                resampled_ds = None
                
                # 清理临时文件
                try:
                    os.remove(temp_input)
                    os.remove(temp_output)
                except:
                    pass
            
            result = out_band.WriteArray(band_data)
            if result != gdal.CE_None:
                raise Exception(f"写入波段{i+1}失败")
            
            out_band.SetColorInterpretation(color_interpretations[i])
            out_band.FlushCache()

        # 强制写入磁盘
        out_ds.FlushCache()
        reference_ds = None
        out_ds = None

        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"合成影像创建完成: {output_path}, 文件大小: {file_size} bytes")
            
            # 验证文件内容
            try:
                verify_ds = gdal.Open(output_path)
                if verify_ds:
                    print(f"验证 - 波段数: {verify_ds.RasterCount}, 尺寸: {verify_ds.RasterXSize}x{verify_ds.RasterYSize}")
                    verify_ds = None
                else:
                    print("警告: 无法验证输出文件")
            except Exception as e:
                print(f"验证输出文件时出错: {str(e)}")
        else:
            raise Exception(f"输出文件未创建: {output_path}")

        return output_path

    def create_ndvi(self, band_paths: Dict[str, str], output_path: str) -> str:
        """
        创建NDVI (归一化植被指数) 影像
        
        Args:
            band_paths: 波段路径字典
            output_path: 输出路径
            
        Returns:
            输出文件路径
        """
        if 'B5' not in band_paths or 'B4' not in band_paths:
            raise Exception("NDVI计算需要B5(NIR)和B4(Red)波段")
        
        # 处理NIR和Red波段
        nir_path = band_paths['B5']
        red_path = band_paths['B4']
        
        # 检查是否是已处理文件并相应处理
        if '_processed' in nir_path or '_clipped' in nir_path:
            print(f"读取已处理NIR波段: {nir_path}")
            dataset = gdal.Open(nir_path)
            nir_reflectance = dataset.ReadAsArray()
            dataset = None
            nir_reflectance = np.clip(nir_reflectance, 0, 1)
        else:
            print(f"处理原始NIR波段: {nir_path}")
            nir_reflectance = self.process_band(nir_path, 'B5', apply_atm_correction=True)
            
        if '_processed' in red_path or '_clipped' in red_path:
            print(f"读取已处理Red波段: {red_path}")
            dataset = gdal.Open(red_path)
            red_reflectance = dataset.ReadAsArray()
            dataset = None
            red_reflectance = np.clip(red_reflectance, 0, 1)
        else:
            print(f"处理原始Red波段: {red_path}")
            red_reflectance = self.process_band(red_path, 'B4', apply_atm_correction=True)
        
        # 计算NDVI: (NIR - Red) / (NIR + Red)
        # 避免除零错误
        denominator = nir_reflectance + red_reflectance
        denominator[denominator == 0] = 0.0001  # 避免除零
        
        ndvi = (nir_reflectance - red_reflectance) / denominator
        
        # 限制NDVI范围在-1到1之间
        ndvi = np.clip(ndvi, -1, 1)
        
        print(f"NDVI统计: min={ndvi.min():.6f}, max={ndvi.max():.6f}, mean={ndvi.mean():.6f}")
        
        # 将NDVI转换为8bit灰度图像
        # NDVI范围通常是-1到1，映射到0-255
        ndvi_8bit = ((ndvi + 1) * 127.5).astype(np.uint8)
        
        print(f"NDVI 8bit统计: min={ndvi_8bit.min()}, max={ndvi_8bit.max()}, mean={ndvi_8bit.mean():.2f}")
        
        # 获取参考数据集
        reference_ds = gdal.Open(band_paths['B4'])
        if reference_ds is None:
            raise Exception("无法打开参考波段文件")
        
        print(f"NDVI输出尺寸: {reference_ds.RasterXSize} x {reference_ds.RasterYSize}")

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path,
                              reference_ds.RasterXSize,
                              reference_ds.RasterYSize,
                              1,
                              gdal.GDT_Byte,
                              options=['COMPRESS=LZW'])
        
        if out_ds is None:
            raise Exception(f"无法创建NDVI输出文件: {output_path}")
        
        out_ds.SetProjection(reference_ds.GetProjection())
        out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
        
        # 确保数据形状匹配
        if ndvi_8bit.shape != (reference_ds.RasterYSize, reference_ds.RasterXSize):
            print(f"NDVI数据形状调整: {ndvi_8bit.shape} -> {(reference_ds.RasterYSize, reference_ds.RasterXSize)}")
            # 使用GDAL调整数据形状
            import tempfile
            temp_input = tempfile.mktemp(suffix='.tif')
            temp_output = tempfile.mktemp(suffix='.tif')
            
            # 保存当前NDVI到临时文件
            temp_driver = gdal.GetDriverByName('GTiff')
            temp_ds = temp_driver.Create(temp_input, ndvi_8bit.shape[1], ndvi_8bit.shape[0], 1, gdal.GDT_Byte)
            temp_ds.GetRasterBand(1).WriteArray(ndvi_8bit)
            temp_ds = None
            
            # 重采样到目标尺寸
            gdal.Warp(temp_output, temp_input, 
                     xSize=reference_ds.RasterXSize, 
                     ySize=reference_ds.RasterYSize, 
                     resampleAlg=gdal.GRA_Bilinear)
            
            # 读取重采样后的数据
            resampled_ds = gdal.Open(temp_output)
            ndvi_8bit = resampled_ds.ReadAsArray()
            resampled_ds = None
            
            # 清理临时文件
            try:
                os.remove(temp_input)
                os.remove(temp_output)
            except:
                pass
        
        # 写入NDVI波段
        out_band = out_ds.GetRasterBand(1)
        if out_band is None:
            raise Exception("无法创建NDVI波段")
        
        result = out_band.WriteArray(ndvi_8bit)
        if result != gdal.CE_None:
            raise Exception("写入NDVI波段失败")
        
        out_band.SetColorInterpretation(gdal.GCI_GrayIndex)
        out_band.FlushCache()
        out_ds.FlushCache()
        
        reference_ds = None
        out_ds = None

        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"NDVI影像创建完成: {output_path}, 文件大小: {file_size} bytes")
        else:
            raise Exception(f"NDVI输出文件未创建: {output_path}")
        
        return output_path

    def one_click_preprocess(self,
                            band_paths: Dict[str, str],
                            output_dir: str,
                            mtl_path: str = None,
                            clip_extent: List[float] = None,
                            clip_shapefile: str = None,
                            create_composites: List[str] = None,
                            apply_cloud_mask: bool = False,
                            qa_band_path: str = None) -> Dict:
        """
        一键完整预处理流程

        Args:
            band_paths: 波段路径字典 {'B1': path, 'B2': path, ...}
            output_dir: 输出目录
            mtl_path: MTL元数据文件路径
            clip_extent: 裁剪范围
            clip_shapefile: 裁剪矢量文件
            create_composites: 要创建的合成影像类型列表
            apply_cloud_mask: 是否应用云掩膜
            qa_band_path: QA波段路径

        Returns:
            处理结果字典
        """
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

            # 读取元数据
            if mtl_path and os.path.exists(mtl_path):
                print(f"正在读取MTL文件: {mtl_path}")
                self.read_mtl_file(mtl_path)
                results['metadata'] = self.metadata
                print(f"MTL元数据读取完成，太阳高度角: {self.metadata.get('sun_elevation', 'N/A')}")
                print(f"辐射定标参数示例 - B4增益: {self.RADIANCE_MULT.get('B4', 'N/A')}, B4偏移: {self.RADIANCE_ADD.get('B4', 'N/A')}")
            else:
                print("警告: 未找到MTL文件，使用默认参数")

            # 处理云掩膜
            cloud_mask = None
            qa_band_ref_path = None
            if apply_cloud_mask and qa_band_path and os.path.exists(qa_band_path):
                try:
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
                except Exception as e:
                    print(f"云掩膜处理失败，跳过云掩膜: {str(e)}")
                    cloud_mask = None
                    qa_band_ref_path = None

            # 处理每个波段
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

            # 创建合成影像
            if create_composites:
                for composite_type in create_composites:
                    composite_path = os.path.join(output_dir, f'{composite_type}.tif')
                    self.create_composite(results['processed_bands'],
                                        composite_path,
                                        composite_type=composite_type)
                    results['composites'][composite_type] = composite_path

            return results

        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            return results