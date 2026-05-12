import sqlite3
import pandas as pd
import os

# 数据库文件路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'heart_disease.db')

def get_connection():
    """获取 SQLite 数据库连接"""
    return sqlite3.connect(DB_PATH)

def init_db(conn):
    """
    初始化数据仓库（星型模型）
    创建维度表和事实表，并在建表前清理旧表以保证可重复执行 (Idempotent)
    """
    cursor = conn.cursor()
    
    # 每次 ETL 都清空旧表，保证系统可以"重新导入数据"
    cursor.execute("DROP TABLE IF EXISTS fact_health_record")
    cursor.execute("DROP TABLE IF EXISTS dim_patient")
    
    # ---------------- 优化项 1：维度表扩充描述字段 ---------------- #
    # 添加了 sex_desc 和 cp_desc，让维度表不仅仅存储数字，
    # 而是真正起到“维度描述”的作用，方便后续直接取用文本进行可视化
    cursor.execute('''
        CREATE TABLE dim_patient (
            patient_key INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            age_group TEXT,
            sex INTEGER,
            sex_desc TEXT,
            cp INTEGER,
            cp_desc TEXT
        )
    ''')
    
    # 构建事实表
    cursor.execute('''
        CREATE TABLE fact_health_record (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_key INTEGER,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            target INTEGER,
            FOREIGN KEY (patient_key) REFERENCES dim_patient(patient_key)
        )
    ''')
    
    # ---------------- 优化项 2：增加索引以提升聚合及查询性能 ---------------- #
    cursor.execute("CREATE INDEX idx_fact_target ON fact_health_record(target)")
    cursor.execute("CREATE INDEX idx_fact_patient_key ON fact_health_record(patient_key)")
    cursor.execute("CREATE INDEX idx_dim_age_group ON dim_patient(age_group)")
    cursor.execute("CREATE INDEX idx_dim_sex ON dim_patient(sex)")
    
    conn.commit()

def generate_dim_patient(df):
    """
    根据清洗后的宽表数据抽取维度表数据，处理业务逻辑并补充描述字段
    """
    dim_df = pd.DataFrame()
    
    # 赋加代理键 (Surrogate Key)
    dim_df['patient_key'] = range(1, len(df) + 1)
    dim_df['age'] = df['age']
    
    # 年龄分箱逻辑
    def get_age_group(age):
        if age < 40:
            return '青年(<40)'
        elif age <= 60:
            return '中年(40-60)'
        else:
            return '老年(>60)'
            
    dim_df['age_group'] = df['age'].apply(get_age_group)
    dim_df['sex'] = df['sex']
    dim_df['sex_desc'] = df['sex'].map({0: '女', 1: '男'})
    dim_df['cp'] = df['cp']
    
    # 胸痛类型文本映射
    cp_map = {
        0: '典型心绞痛(Typical)', 
        1: '非典型心绞痛(Atypical)', 
        2: '非心绞痛(Non-anginal)', 
        3: '无症状(Asymptomatic)'
    }
    dim_df['cp_desc'] = df['cp'].map(cp_map).fillna('未知(Unknown)')
    
    return dim_df

def etl_to_sqlite(cleaned_df):
    """
    ETL 主函数：将 Pandas DataFrame 转换为星型模型结构并写入 SQLite 库
    """
    conn = get_connection()
    try:
        # 1. 建表初始化
        init_db(conn)
        
        # 2. 转换 (Transform): 构建维度表数据
        dim_patient_df = generate_dim_patient(cleaned_df)
        
        # 3. 转换 (Transform): 构建事实表数据
        # 事实表继承宽表的大部分特征，并连接维度表的外键
        fact_df = cleaned_df.copy()
        fact_df.insert(0, 'patient_key', dim_patient_df['patient_key'])
        
        # 4. 加载 (Load): 使用 Pandas 的 to_sql 进行批量快速插入
        # 通过 if_exists='append' 和 index=False 将数据直接装载，自增键由 SQLite 自动接管
        dim_patient_df.to_sql('dim_patient', conn, if_exists='append', index=False)
        fact_df.to_sql('fact_health_record', conn, if_exists='append', index=False)
        
        total_records = len(fact_df)
        print(f"[ETL Success] 成功处理并入库 {total_records} 条事实记录。")
        return total_records
        
    except Exception as e:
        print(f"[ETL Error] 数据库加载失败: {e}")
        raise
    finally:
        conn.close()