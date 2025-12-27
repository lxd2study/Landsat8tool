"""文件管理服务"""

import os
import tempfile
import shutil
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List, Dict


class FileManager:
    """文件管理器"""

    def __init__(self):
        self.temp_dirs: Dict[str, str] = {}
        self.lock = threading.Lock()

    def create_temp_dir(self, prefix: str = "landsat8_") -> str:
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        with self.lock:
            self.temp_dirs[temp_dir] = temp_dir
        return temp_dir

    def cleanup_temp_dir(self, temp_dir: str):
        """清理临时目录"""
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理临时目录失败: {str(e)}")
        with self.lock:
            self.temp_dirs.pop(temp_dir, None)

    def save_uploaded_file(self, file_content: bytes, target_dir: str, filename: str) -> str:
        """保存上传文件"""
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return file_path

    def save_band_files(self, band_files, temp_band_dir: str) -> Dict[str, str]:
        """保存波段文件并返回波段路径映射"""
        band_paths = {}
        for band_file in band_files:
            filename = band_file.filename

            # 从文件名中提取波段编号
            band_name = None
            for i in range(1, 12):
                if f'B{i}.' in filename.upper() or f'_B{i}.' in filename.upper():
                    band_name = f'B{i}'
                    break

            if not band_name:
                print(f"无法从文件名中识别波段编号: {filename}，跳过该文件")
                continue

            file_extension = os.path.splitext(filename)[-1]
            temp_band_path = os.path.join(temp_band_dir, f"{band_name}{file_extension}")

            content = band_file.file.read()
            with open(temp_band_path, 'wb') as f:
                f.write(content)

            band_paths[band_name] = temp_band_path

        return band_paths

    def save_shapefiles(self, shape_files, temp_shape_dir: str) -> Optional[str]:
        """保存裁剪矢量文件"""
        shapefile_path = None
        for shape_file in shape_files:
            filename = shape_file.filename
            file_extension = os.path.splitext(filename)[-1].lower()

            if file_extension not in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
                print(f"不支持的矢量文件类型: {file_extension}，文件名: {filename}")
                continue

            temp_shape_path = os.path.join(temp_shape_dir, filename)
            content = shape_file.file.read()
            with open(temp_shape_path, 'wb') as f:
                f.write(content)

            if file_extension == '.shp':
                shapefile_path = temp_shape_path

        return shapefile_path

    def parse_extent(self, extent_str: str) -> Optional[List[float]]:
        """解析裁剪范围字符串"""
        if not extent_str:
            return None

        try:
            extent_list = [float(x.strip()) for x in extent_str.split(',')]
            if len(extent_list) != 4:
                raise ValueError("裁剪范围必须包含4个值: xmin,ymin,xmax,ymax")
            return extent_list
        except Exception as e:
            raise ValueError(f"裁剪范围格式错误: {str(e)}")

    def parse_composites(self, composite_str: str) -> Optional[List[str]]:
        """解析合成类型字符串"""
        if not composite_str:
            return None

        composite_list = [c.strip() for c in composite_str.split(',') if c.strip()]
        valid_composites = ['true_color', 'false_color', 'natural_color', 'agriculture', 'urban', 'swir', 'ndvi']

        for comp in composite_list:
            if comp not in valid_composites:
                raise ValueError(f"不支持的合成类型: {comp}。有效类型: {', '.join(valid_composites)}")

        return composite_list

    def select_folder_dialog(self) -> Optional[str]:
        """打开文件夹选择对话框"""
        selected_path = None
        error_message = None

        def open_dialog():
            nonlocal selected_path, error_message
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                folder_path = filedialog.askdirectory(
                    title="选择输出文件夹",
                    initialdir=os.path.expanduser("~")
                )

                if folder_path:
                    selected_path = folder_path

                root.destroy()
            except Exception as e:
                error_message = str(e)

        # 在新线程中运行对话框
        dialog_thread = threading.Thread(target=open_dialog)
        dialog_thread.start()
        dialog_thread.join(timeout=60)

        if error_message:
            raise Exception(f"打开文件选择对话框失败: {error_message}")

        return selected_path
