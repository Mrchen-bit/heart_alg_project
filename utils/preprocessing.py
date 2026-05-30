import pandas as pd
import numpy as np
import os

def clean_heart_data(input_path, output_path):
    """
    数据清洗流程：
    1. 缺失值用中位数填充
    2. 异常值处理（静息血压trestbps: 80-200，胆固醇chol: 100-600）
    3. 类别特征整数编码（sex, cp, restecg, slope, thal）
    4. 保存清洗后数据
    5. 检查完整性
    """
    df = pd.read_csv(input_path)

    # 1. 缺失值用中位数填充
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # 2. 异常值处理
    # 静息血压（trestbps）范围 80-200
    df.loc[(df['trestbps'] < 80) | (df['trestbps'] > 200), 'trestbps'] = df['trestbps'].median()
    # 胆固醇（chol）范围 100-600
    df.loc[(df['chol'] < 100) | (df['chol'] > 600), 'chol'] = df['chol'].median()

    # 3. 类别特征整数编码
    cat_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    # 4. 保存清洗后数据
    df.to_csv(output_path, index=False)

    # 5. 检查完整性
    assert not df.isnull().any().any(), '存在空值！'
    assert df['trestbps'].between(80, 200).all(), 'trestbps超出范围！'
    assert df['chol'].between(100, 600).all(), 'chol超出范围！'
    print('数据清洗完成，完整性检查通过。')

if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
    input_path = os.path.join(DATA_DIR, 'heart.csv')
    output_path = os.path.join(DATA_DIR, 'cleaned_heart.csv')
    clean_heart_data(input_path, output_path)
