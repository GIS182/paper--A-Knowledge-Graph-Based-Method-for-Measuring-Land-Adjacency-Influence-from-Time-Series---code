import sqlite3

def create_gpkg_spatial_index(gpkg_path: str, layer_name: str):
    """
    为GPKG图层创建R树空间索引
    参数:
        gpkg_path: GeoPackage文件路径
        layer_name: 目标图层名称
    原理:
        执行SQL: SELECT CreateSpatialIndex('图层名', 'geometry')
    """
    try:
        with sqlite3.connect(gpkg_path) as conn:
            cursor = conn.cursor()

            # 检查是否已有空间索引
            table_name = f"idx_{layer_name}_geometry"  # 分离变量
            cursor.execute("""
                SELECT 1 FROM sqlite_master 
                WHERE type='table' AND name='?'
            """, (table_name,))
            if cursor.fetchone():
                print(f"⚠️ 空间索引已存在: {layer_name}")
                return

            # 创建R树索引[6,9]
            cursor.execute(f"SELECT CreateSpatialIndex('{layer_name}', 'geometry')")
            conn.commit()
            print(f"✅ 空间索引创建成功: {layer_name}")

    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            raise ValueError(f"图层不存在: {layer_name}")
        raise RuntimeError(f"索引创建失败: {str(e)}")