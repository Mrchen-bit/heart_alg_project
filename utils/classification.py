# -*- coding: utf-8 -*-
import os
import pandas as pd
from typing import List, Union, Dict, Any
from sklearn.ensemble import RandomForestClassifier

# 训练数据路径（优先用cleaned_heart.csv）
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cleaned_heart.csv")

FEATURE_COLS = [
	"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
	"exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

def load_train_data(path: str = DATA_PATH) -> pd.DataFrame:
	"""加载训练数据。"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"训练数据文件不存在: {path}")
	df = pd.read_csv(path)
	# 只保留需要的特征和目标
	cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
	return df[cols].dropna()

def train_rf_model(df: pd.DataFrame) -> RandomForestClassifier:
	"""训练随机森林分类器。"""
	X = df[FEATURE_COLS]
	y = df[TARGET_COL]
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X, y)
	return model

def predict_risk(
	input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
) -> pd.DataFrame:
	"""
	输入:
		- DataFrame: 必须包含所有特征列
		- dict 或 list[dict]: 自动转为DataFrame
	输出:
		- DataFrame: 包含预测类别和概率
	"""
	# 加载并训练模型
	train_df = load_train_data()
	model = train_rf_model(train_df)

	# 处理输入
	if isinstance(input_data, dict):
		input_df = pd.DataFrame([input_data])
	elif isinstance(input_data, list):
		input_df = pd.DataFrame(input_data)
	elif isinstance(input_data, pd.DataFrame):
		input_df = input_data.copy()
	else:
		raise ValueError("输入数据格式不支持")

	# 检查特征列
	missing = [col for col in FEATURE_COLS if col not in input_df.columns]
	if missing:
		raise ValueError(f"输入数据缺少特征列: {missing}")

	X_pred = input_df[FEATURE_COLS]
	pred_label = model.predict(X_pred)
	pred_prob = model.predict_proba(X_pred)[:, 1]  # 预测为1的概率

	result = input_df.copy()
	result["pred_label"] = pred_label
	result["pred_prob"] = pred_prob
	return result

# Streamlit集成接口
def streamlit_predict_form():
	"""在Streamlit中渲染预测表单并返回预测结果。"""
	import streamlit as st
	user_input = {}
	col1, col2, col3 = st.columns(3)
	with col1:
		user_input["age"] = st.number_input("年龄 (Age)", min_value=1, max_value=120, value=50)
		user_input["sex"] = st.selectbox("性别 (Sex)", options=[1, 0], format_func=lambda x: "男 (1)" if x == 1 else "女 (0)")
		user_input["cp"] = st.selectbox("胸痛类型 (CP)", options=[0, 1, 2, 3])
		user_input["trestbps"] = st.number_input("静息血压 (Trestbps)", value=120)
	with col2:
		user_input["chol"] = st.number_input("胆固醇 (Chol)", value=200)
		user_input["fbs"] = st.selectbox("空腹血糖 > 120 mg/dl (FBS)", options=[0, 1])
		user_input["restecg"] = st.selectbox("静息心电图 (RestECG)", options=[0, 1, 2])
		user_input["thalach"] = st.number_input("最大心率 (Thalach)", value=150)
	with col3:
		user_input["exang"] = st.selectbox("运动引发心绞痛 (Exang)", options=[0, 1])
		user_input["oldpeak"] = st.number_input("ST段压低 (Oldpeak)", value=1.0, step=0.1)
		user_input["slope"] = st.selectbox("ST段斜率 (Slope)", options=[0, 1, 2])
		user_input["ca"] = st.selectbox("主要血管数 (CA)", options=[0, 1, 2, 3, 4])
		user_input["thal"] = st.selectbox("地中海贫血 (Thal)", options=[0, 1, 2, 3])
	submit = st.button("开始智能预测", use_container_width=True)
	if submit:
		with st.spinner("AI模型正在评估..."):
			result = predict_risk(user_input)
			prob = result.iloc[0]["pred_prob"]
			label = result.iloc[0]["pred_label"]
			st.markdown("### 评估结果")
			if prob > 0.5:
				st.error(f"高风险警告！预测患病概率为 {prob:.1%}")
				st.toast('发现高风险指标，请注意防范！')
			else:
				st.success(f"状况良好，患病概率为 {prob:.1%}")
				st.balloons()
			st.progress(prob)
			st.write("预测类别:", "高风险" if label == 1 else "低风险")
