"""几何处理操作模块"""

import os
import tempfile
import numpy as np
from osgeo import gdal
from typing import List, Optional


def clip_raster(input_path: str, output_path: str,
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


def resample_to_match(input_path: str, target_path: str,
                     output_path: str = None,
                     resample_alg: str = 'bilinear') -> str:
    """
    重采样到目标影像的分辨率和范围

    Args:
        input_path: 输入影像路径
        target_path: 目标参考影像路径
        output_path: 输出路径，如果为None则返回临时文件路径
        resample_alg: 重采样算法 ('bilinear', 'nearest')

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.tif')

    target_ds = gdal.Open(target_path)
    if target_ds is None:
        raise Exception(f"无法打开目标影像: {target_path}")

    alg_map = {
        'bilinear': gdal.GRA_Bilinear,
        'nearest': gdal.GRA_NearestNeighbour
    }

    gdal.Warp(output_path, input_path,
             width=target_ds.RasterXSize,
             height=target_ds.RasterYSize,
             resampleAlg=alg_map.get(resample_alg, gdal.GRA_Bilinear),
             dstNodata=0)

    target_ds = None
    return output_path


def pansharpening(pan_path: str, multi_paths: List[str],
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
