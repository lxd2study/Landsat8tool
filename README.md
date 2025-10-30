# Landsat 8 影像预处理工具

一个基于 FastAPI 和 Vue.js 的 Web 应用程序，提供完整的 Landsat 8 影像预处理功能，包括辐射定标、大气校正、云掩膜处理、影像裁剪和波段合成。

## 🚀 主要功能

- **辐射定标**: DN值 → 辐射亮度 → 地表反射率转换
- **大气校正**: DOS暗目标法大气校正
- **云掩膜处理**: 基于QA波段的云检测和掩膜
- **影像裁剪**: 支持按范围或矢量文件裁剪
- **波段合成**: 多种预设合成方案（真彩色、假彩色、农业、城市等）
- **NDVI计算**: 归一化植被指数计算
- **批处理**: 一键完成所有预处理步骤

## 📋 系统要求

### 运行环境
- Python 3.8+
- Windows/Linux/MacOS

### 依赖库
```
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
numpy>=1.21.0
gdal>=3.3.0
```

## 🛠️ 安装与部署

### 1. 克隆项目
```bash
git clone https://github.com/lxd2study/Landsat8tool.git
cd Landsat8-v1.0.0
```

### 2. 创建虚拟环境
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/MacOS
source .venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

如果没有 requirements.txt，请手动安装：
```bash
pip install fastapi uvicorn python-multipart numpy gdal
```

### 4. 启动服务
```bash
python main.py
```

服务将在 `http://localhost:5001` 启动

### 5. 访问应用
打开浏览器访问 `http://localhost:5001` 即可使用Web界面

## 📖 使用指南

### 1. 准备数据
确保您有以下Landsat 8数据文件：
- 波段文件：`LC08_*.B1.TIF`, `LC08_*.B2.TIF`, ..., `LC08_*.B7.TIF`
- MTL元数据文件：`LC08_*_MTL.txt`（可选，推荐提供）
- QA波段文件：`LC08_*_BQA.TIF`（可选，用于云掩膜）

### 2. 上传文件
1. 在Web界面中，将波段文件拖拽到上传区域
2. 系统会自动识别波段编号（B1-B11）
3. 可选择上传MTL和QA文件以获得更精确的处理结果

### 3. 配置参数
- **输出目录**: 选择处理结果的保存位置
- **合成影像类型**: 选择要生成的合成影像（真彩色、假彩色、NDVI等）
- **云掩膜**: 如果上传了QA文件，可选择应用云掩膜

### 4. 开始处理
点击"开始预处理"按钮，系统将自动完成所有处理步骤：
- 文件上传和验证
- 元数据解析
- 辐射定标
- 大气校正
- 云掩膜处理（如果启用）
- 影像裁剪（如果设置了范围）
- 波段合成
- 结果保存

### 5. 查看结果
处理完成后，您可以在输出目录中找到：
- `B1_processed.tif` - `B7_processed.tif`: 处理后的单波段影像
- `true_color.tif`: 真彩色合成影像
- `false_color.tif`: 假彩色合成影像
- `ndvi.tif`: NDVI植被指数影像
- `cloud_mask.tif`: 云掩膜文件（如果启用）

## 🎨 波段合成类型

| 类型   | 波段组合     | 描述                  | 应用场景           |
|------|----------|---------------------|----------------|
| 真彩色  | B4-B3-B2 | Red-Green-Blue      | 自然地物识别，最接近人眼视觉 |
| 假彩色  | B5-B4-B3 | NIR-Red-Green       | 植被健康监测，植物呈红色   |
| 农业   | B6-B5-B2 | SWIR1-NIR-Blue      | 农作物类型识别，土壤湿度评估 |
| 城市   | B7-B6-B4 | SWIR2-SWIR1-Red     | 建筑物识别，城市规划     |
| 自然彩色 | B5-B4-B3 | NIR-Red-Green       | 地形地貌分析         |
| 短波红外 | B7-B5-B3 | SWIR2-NIR-Green     | 水体识别，云雾穿透      |
| NDVI | B5-B4    | (NIR-Red)/(NIR+Red) | 植被覆盖度，植被健康状况   |

## 📊 Landsat 8 波段信息

| 波段  | 名称     | 波长范围          | 分辨率  | 主要用途         |
|-----|--------|---------------|------|--------------|
| B1  | 海岸/气溶胶 | 0.43-0.45 μm  | 30m  | 浅水映射，气溶胶研究   |
| B2  | 蓝      | 0.45-0.51 μm  | 30m  | 水体穿透，土壤/植被识别 |
| B3  | 绿      | 0.53-0.59 μm  | 30m  | 植被活力评估       |
| B4  | 红      | 0.64-0.67 μm  | 30m  | 植被分类，生物量估算   |
| B5  | 近红外    | 0.85-0.88 μm  | 30m  | 生物量，海岸线映射    |
| B6  | 短波红外1  | 1.57-1.65 μm  | 30m  | 土壤湿度，植被含水量   |
| B7  | 短波红外2  | 2.11-2.29 μm  | 30m  | 岩石/矿物识别，云雾穿透 |
| B8  | 全色     | 0.50-0.68 μm  | 15m  | 高分辨率影像锐化     |
| B9  | 卷云     | 1.36-1.38 μm  | 30m  | 卷云检测         |
| B10 | 热红外1   | 10.6-11.19 μm | 100m | 地表温度         |
| B11 | 热红外2   | 11.5-12.51 μm | 100m | 地表温度         |

## 🔧 工具箱功能

### MTL元数据解析
- 上传MTL文件查看影像元数据
- 显示场景ID、采集日期、太阳角度等信息
- 自动提取辐射定标参数

### 云掩膜分析
- 分析QA波段中的云信息
- 可设置置信度阈值（低/中/高）
- 显示云覆盖率统计

## 📡 API 接口

服务提供以下REST API接口：

- `POST /preprocess_landsat8` - 一键预处理
- `GET /composite_types` - 获取合成类型
- `GET /band_info` - 获取波段信息
- `POST /read_mtl` - 解析MTL文件
- `POST /extract_cloud_mask` - 提取云掩膜
- `GET /select_folder` - 选择输出文件夹
- `GET /health` - 健康检查

API文档可在 `http://localhost:5001/docs` 查看

## 🚨 注意事项

1. **文件命名**: 确保波段文件名包含波段编号（如B1, B2等）
2. **内存使用**: 处理大影像时需要足够的内存
3. **GDAL安装**: 在某些系统上可能需要单独安装GDAL库
4. **文件格式**: 目前支持 .tif, .tiff, .img 格式
5. **输出路径**: 确保输出目录有足够的磁盘空间

## 🐛 常见问题

### Q: GDAL安装失败怎么办？
A: 可以尝试使用conda安装：`conda install -c conda-forge gdal`

### Q: 处理大文件时内存不足？
A: 可以尝试分批处理，或者增加系统虚拟内存

### Q: 云掩膜不准确？
A: 尝试调整置信度阈值，或检查QA波段文件是否正确

### Q: 合成影像颜色异常？
A: 检查波段文件是否正确对应，确保MTL文件提供了准确的定标参数

## 📝 开发信息

- **版本**: 1.0.0
- **开发者**: LXD
- **框架**: FastAPI + Vue.js + Element Plus
- **许可证**: MIT

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: https://github.com/lxd2study/Landsat8tool/issues

---

© 2025 LXD | Landsat8批量处理工具 v1.0.0 | 基于FastAPI框架开发
