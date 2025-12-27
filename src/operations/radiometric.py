"""辐射定标操作模块"""

import numpy as np
from typing import Dict


def dn_to_radiance(dn_array: np.ndarray, band_name: str,
                   radiance_mult: Dict, radiance_add: Dict) -> np.ndarray:
    """
    DN值转辐射亮度

    公式: L = ML * DN + AL
    其中: L-辐射亮度, ML-增益, AL-偏移, DN-像元值

    Args:
        dn_array: DN值数组
        band_name: 波段名称 (如 'B1', 'B2')
        radiance_mult: 辐射增益参数字典
        radiance_add: 辐射偏移参数字典

    Returns:
        辐射亮度数组
    """
    ml = radiance_mult.get(band_name, 0.00001)
    al = radiance_add.get(band_name, 0.1)

    radiance = ml * dn_array.astype(np.float32) + al

    # 确保辐射亮度为正值
    radiance = np.maximum(radiance, 0.0)

    return radiance


def radiance_to_reflectance(radiance: np.ndarray, band_name: str,
                           esun: Dict, sun_elevation: float = 45.0) -> np.ndarray:
    """
    辐射亮度转地表反射率 (TOA反射率)

    公式: ρ = (π * L * d²) / (ESUN * sin(θ))
    其中: ρ-反射率, L-辐射亮度, d-日地距离, ESUN-太阳辐照度, θ-太阳高度角

    Args:
        radiance: 辐射亮度数组
        band_name: 波段名称
        esun: 太阳辐照度参数字典
        sun_elevation: 太阳高度角(度)

    Returns:
        反射率数组 (0-1范围)
    """
    # 日地距离校正因子 (简化处理)
    d = 1.0

    # 太阳辐照度
    esun_value = esun.get(band_name, 1500.0)

    # 太阳高度角转弧度
    sun_elevation_rad = np.deg2rad(sun_elevation)

    # 计算反射率
    reflectance = (np.pi * radiance * d * d) / (esun_value * np.sin(sun_elevation_rad))

    # 限制在合理范围，但允许小的负值用于后续处理
    reflectance = np.clip(reflectance, -0.1, 2.0)

    return reflectance
