"""波段合成操作模块"""

import os
import tempfile
import numpy as np
from osgeo import gdal
from typing import Dict, List
from ..core.constants import COMPOSITE_MAP


def create_composite(band_paths: Dict[str, str],
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
    if composite_type not in COMPOSITE_MAP:
        raise Exception(f"不支持的合成类型: {composite_type}")

    bands_to_use = COMPOSITE_MAP[composite_type]

    # 特殊处理NDVI
    if composite_type == 'ndvi':
        return create_ndvi(band_paths, output_path)

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
            dataset = gdal.Open(band_path)
            if dataset is None:
                raise Exception(f"无法打开已处理波段文件: {band_path}")

            reflectance = dataset.ReadAsArray()
            dataset = None

            # 确保数据在合理范围内
            reflectance = np.clip(reflectance, 0, 1)
        else:
            # 如果是原始文件，需要先处理（这里假设外部已处理）
            raise Exception(f"波段 {band_name} 需要先进行预处理: {band_path}")

        # 检查数组形状
        band_shapes.append(reflectance.shape)

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
        target_shape = band_shapes[0]
        target_width, target_height = target_shape[1], target_shape[0]

        for i in range(len(processed_bands)):
            if processed_bands[i].shape != target_shape:
                # 使用GDAL进行重采样
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

    # 验证参考数据集
    if reference_ds is None:
        raise Exception("参考数据集为空")

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

    # 写入波段
    color_interpretations = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand]

    for i, band_data in enumerate(processed_bands):
        out_band = out_ds.GetRasterBand(i + 1)

        # 确保数据形状匹配
        if band_data.shape != (reference_ds.RasterYSize, reference_ds.RasterXSize):
            # 使用GDAL调整数据形状
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

        out_band.WriteArray(band_data)
        out_band.SetColorInterpretation(color_interpretations[i])
        out_band.FlushCache()

    out_ds.FlushCache()
    reference_ds = None
    out_ds = None

    return output_path


def create_ndvi(band_paths: Dict[str, str], output_path: str) -> str:
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
        dataset = gdal.Open(nir_path)
        nir_reflectance = dataset.ReadAsArray()
        dataset = None
        nir_reflectance = np.clip(nir_reflectance, 0, 1)
    else:
        raise Exception(f"NIR波段需要先进行预处理: {nir_path}")

    if '_processed' in red_path or '_clipped' in red_path:
        dataset = gdal.Open(red_path)
        red_reflectance = dataset.ReadAsArray()
        dataset = None
        red_reflectance = np.clip(red_reflectance, 0, 1)
    else:
        raise Exception(f"Red波段需要先进行预处理: {red_path}")

    # 计算NDVI: (NIR - Red) / (NIR + Red)
    # 避免除零错误
    denominator = nir_reflectance + red_reflectance
    denominator[denominator == 0] = 0.0001  # 避免除零

    ndvi = (nir_reflectance - red_reflectance) / denominator

    # 限制NDVI范围在-1到1之间
    ndvi = np.clip(ndvi, -1, 1)

    # 将NDVI转换为8bit灰度图像
    # NDVI范围通常是-1到1，映射到0-255
    ndvi_8bit = ((ndvi + 1) * 127.5).astype(np.uint8)

    # 获取参考数据集
    reference_ds = gdal.Open(band_paths['B4'])
    if reference_ds is None:
        raise Exception("无法打开参考波段文件")

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
        # 使用GDAL调整数据形状
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
    out_band.WriteArray(ndvi_8bit)
    out_band.SetColorInterpretation(gdal.GCI_GrayIndex)
    out_band.FlushCache()
    out_ds.FlushCache()

    reference_ds = None
    out_ds = None

    return output_path
