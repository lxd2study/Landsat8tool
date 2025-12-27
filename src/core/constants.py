"""常量定义模块"""

# Landsat 8 OLI 辐射定标参数
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

# 波段合成映射
COMPOSITE_MAP = {
    'true_color': ['B4', 'B3', 'B2'],      # 真彩色: Red-Green-Blue
    'false_color': ['B5', 'B4', 'B3'],     # 假彩色: NIR-Red-Green
    'agriculture': ['B6', 'B5', 'B2'],     # 农业: SWIR1-NIR-Blue
    'urban': ['B7', 'B6', 'B4'],           # 城市: SWIR2-SWIR1-Red
    'natural_color': ['B5', 'B4', 'B3'],   # 自然彩色: NIR-Red-Green
    'swir': ['B7', 'B5', 'B3'],            # 短波红外: SWIR2-NIR-Green
    'ndvi': ['B5', 'B4'],                  # NDVI: NIR-Red (特殊处理)
}

# 波段信息
BAND_INFO = {
    "satellite": "Landsat 8",
    "sensor": "OLI/TIRS",
    "bands": [
        {"band": "B1", "name": "海岸/气溶胶", "wavelength": "0.43-0.45 μm", "resolution": "30m", "use": "浅水映射，气溶胶研究"},
        {"band": "B2", "name": "蓝", "wavelength": "0.45-0.51 μm", "resolution": "30m", "use": "水体穿透，土壤/植被识别"},
        {"band": "B3", "name": "绿", "wavelength": "0.53-0.59 μm", "resolution": "30m", "use": "植被活力评估"},
        {"band": "B4", "name": "红", "wavelength": "0.64-0.67 μm", "resolution": "30m", "use": "植被分类，生物量估算"},
        {"band": "B5", "name": "近红外", "wavelength": "0.85-0.88 μm", "resolution": "30m", "use": "生物量，海岸线映射"},
        {"band": "B6", "name": "短波红外1", "wavelength": "1.57-1.65 μm", "resolution": "30m", "use": "土壤湿度，植被含水量"},
        {"band": "B7", "name": "短波红外2", "wavelength": "2.11-2.29 μm", "resolution": "30m", "use": "岩石/矿物识别，云雾穿透"},
        {"band": "B8", "name": "全色", "wavelength": "0.50-0.68 μm", "resolution": "15m", "use": "高分辨率影像锐化"},
        {"band": "B9", "name": "卷云", "wavelength": "1.36-1.38 μm", "resolution": "30m", "use": "卷云检测"},
        {"band": "B10", "name": "热红外1", "wavelength": "10.6-11.19 μm", "resolution": "100m", "use": "地表温度"},
        {"band": "B11", "name": "热红外2", "wavelength": "11.5-12.51 μm", "resolution": "100m", "use": "地表温度"}
    ]
}

# 进度步骤定义
PROGRESS_STEPS = [
    {"id": "upload", "title": "上传文件", "detail": "保存上传数据到服务器"},
    {"id": "prepare_output", "title": "准备输出目录", "detail": "创建输出路径"},
    {"id": "metadata", "title": "解析元数据", "detail": "读取 MTL 参数"},
    {"id": "cloud_mask", "title": "云掩膜处理", "detail": "提取并写出云掩膜"},
    {"id": "bands", "title": "波段处理", "detail": "辐射定标与大气校正"},
    {"id": "clip", "title": "影像裁剪", "detail": "按范围或矢量裁剪"},
    {"id": "composite", "title": "波段合成", "detail": "生成合成影像"},
    {"id": "finalize", "title": "结果写入", "detail": "写入输出目录"},
]
