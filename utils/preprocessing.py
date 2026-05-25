import pandas as pd
import numpy as np
import os

# ===================== 自动获取正确路径 =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
csv_path = os.path.join(root_dir, "data", "heart.csv")

print("正在读取文件：", csv_path)

# ===================== 1. 加载数据 =====================
print("===== 开始加载心脏病数据集 =====")
df = pd.read_csv(csv_path)
print(f"原始数据形状: {df.shape}")

# ===================== 2. 缺失值处理（中位数填充） =====================
print("\n===== 处理缺失值 =====")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("缺失值处理完成 ✅")

# ===================== 3. 异常值处理 =====================
print("\n===== 处理异常值 =====")
# 静息血压 60~200
bp_median = df["trestbps"].median()
df.loc[(df["trestbps"] < 60) | (df["trestbps"] > 200), "trestbps"] = bp_median

# 胆固醇 100~600
chol_median = df["chol"].median()
df.loc[(df["chol"] < 100) | (df["chol"] > 600), "chol"] = chol_median
print("异常值处理完成 ✅")

# ===================== 4. 类别特征整数编码 =====================
print("\n===== 类别特征整数编码 =====")
cat_cols = ["sex", "cp", "restecg", "slope", "thal"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes
print("类别编码完成 ✅")

# ===================== 5. 保存清洗后数据 =====================
output_csv = os.path.join(root_dir, "data", "cleaned_heart.csv")
df.to_csv(output_csv, index=False)
print("\n已生成：cleaned_heart.csv ✅")

# ===================== 6. 数据完整性测试 =====================
print("\n===== 数据完整性测试 =====")
test_results = []

test1 = "无空值"
pass1 = df.isnull().sum().sum() == 0
test_results.append([test1, "通过" if pass1 else "失败"])

test2 = "血压 60~200"
pass2 = ((df["trestbps"] >= 60) & (df["trestbps"] <= 200)).all()
test_results.append([test2, "通过" if pass2 else "失败"])

test3 = "胆固醇 100~600"
pass3 = ((df["chol"] >= 100) & (df["chol"] <= 600)).all()
test_results.append([test3, "通过" if pass3 else "失败"])

test4 = "类别特征已整数编码"
pass4 = True
for col in cat_cols:
    if df[col].dtype not in ["int64", "int32", "int8"]:
        pass4 = False
test_results.append([test4, "通过" if pass4 else "失败"])

test_df = pd.DataFrame(test_results, columns=["测试项", "结果"])
print(test_df)

# ===================== 7. 生成测试报告（不使用tabulate，彻底免安装） =====================
report_path = os.path.join(root_dir, "data", "preprocessing_test_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# 心脏病数据预处理测试报告\n\n")
    f.write(f"- 原始数据形状：{df.shape}\n")
    f.write("- 缺失值处理：中位数填充\n")
    f.write("- 异常值处理：血压60-200、胆固醇100-600，中位数修正\n")
    f.write("- 编码特征：sex, cp, restecg, slope, thal\n\n")
    f.write("## 测试结果\n")
    # 纯文本输出，不需要任何包
    f.write("| 测试项 | 结果 |\n")
    f.write("|--------|------|\n")
    for item, res in test_results:
        f.write(f"| {item} | {res} |\n")
    f.write("\n## 结论：数据预处理完成，可用于建模\n")

print("\n=====================================")
print("✅ 全部任务完成！")
print(f"✅ 清洗后数据：{output_csv}")
print(f"✅ 测试报告：{report_path}")
print("=====================================")