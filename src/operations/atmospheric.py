"""大气校正操作模块"""

import numpy as np
from osgeo import gdal


def dark_object_subtraction(reflectance: np.ndarray,
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


def cloud_mask_from_qa(qa_band_path: str,
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
