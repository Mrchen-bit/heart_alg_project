import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="心脏病智能分析与预测", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = "data"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_heart.csv")
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data
def load_data():
    if os.path.exists(CLEANED_DATA_PATH):
        df = pd.read_csv(CLEANED_DATA_PATH)
        df.columns = df.columns.str.lower()
        return df
    return None

@st.cache_resource
def train_model(df):
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[feature_cols]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, feature_cols

def save_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    df.dropna(inplace=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return df

def aggregate_by_age(df):
    bins = [0, 40, 50, 60, 120]
    labels = ['<40', '40-49', '50-59', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    age_risk = df.groupby('age_group')['target'].mean().reset_index()
    age_risk.columns = ['年龄段', '患病率']
    return age_risk

def gender_risk(df):
    gender_risk = df.groupby('sex')['target'].mean().reset_index()
    gender_risk['sex'] = gender_risk['sex'].map({1: '男性', 0: '女性'})
    return gender_risk

def chest_pain_risk(df):
    cp_risk = df.groupby('cp')['target'].mean().reset_index()
    cp_risk['cp'] = cp_risk['cp'].map({0: '典型心绞痛', 1: '非典型心绞痛', 2: '非心绞痛', 3: '无症状'})
    return cp_risk

def run_apriori_rules(df, min_support, min_confidence):
    df_temp = df.copy()
    df_temp['age>50'] = (df_temp['age'] > 50).astype(int)
    df_temp['high_bp'] = (df_temp['trestbps'] > 140).astype(int)
    df_temp['high_chol'] = (df_temp['chol'] > 240).astype(int)
    df_temp['chest_pain_risk'] = (df_temp['cp'].isin([2,3])).astype(int)
    df_temp['target_1'] = df_temp['target']
    feature_cols = ['age>50', 'high_bp', 'high_chol', 'chest_pain_risk', 'target_1']
    basket = df_temp[feature_cols]
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if len(rules) == 0:
        return pd.DataFrame()
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    result = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    result = result[result['consequents'].str.contains('target_1')]
    return result

def main():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #1e293b;'>医疗数据中台</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("数据接入与预处理")
        uploaded_file = st.file_uploader("上传 CSV 数据集", type=['csv'])
        if st.button("重新导入与清洗 (ETL)", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("正在加载、清洗并写入数据库..."):
                    df = save_uploaded_data(uploaded_file)
                    st.success(f"数据处理完毕！共 {len(df)} 条记录已入库。")
                    st.rerun()
            else:
                st.error("请先上传 CSV 文件。")
        st.markdown("---")
        st.caption("当前数据集：cleaned_heart.csv（若存在）")
        # 提供备用渲染选项，解决图表不显示问题
        use_html_fallback = st.checkbox("使用 HTML 渲染（解决图表不显示问题）", value=False)

    st.markdown("<h1 style='color: #1e293b;'>心脏病数据仓库与智能挖掘系统</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #334155; font-size: 1.1rem;'>基于多维数据建模，支持高级聚合查询、Apriori 关联规则挖掘以及机器学习风险预测。</p>", unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.warning("⚠️ 未找到清洗后的数据文件。请通过侧边栏上传 CSV 文件并点击“重新导入与清洗”。")
        st.stop()

    total = len(df)
    prevalence = df['target'].mean() * 100
    avg_age = df['age'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("当前数据总量", f"{total} 条")
    col2.metric("总体患病率", f"{prevalence:.1f}%")
    col3.metric("患者平均年龄", f"{avg_age:.0f} 岁")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 聚合分析与可视化", "🔗 关联规则挖掘", "🤖 风险预测模型"])

    with tab1:
        st.subheader("多维数据集计与透视")
        st.markdown("通过数据仓库的多维模型展示关键特征对心脏病风险的影响。")
        analysis_type = st.selectbox("选择聚合维度", ["年龄段患病率", "性别患病率", "胸痛类型患病率", "自定义交叉分析"])
        if analysis_type == "年龄段患病率":
            age_risk = aggregate_by_age(df)
            fig = px.bar(age_risk, x='年龄段', y='患病率', title="不同年龄段患病率分布", color='患病率', color_continuous_scale='Blues')
            fig.update_layout(template='simple_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif analysis_type == "性别患病率":
            gender_risk_df = gender_risk(df)
            fig = px.bar(gender_risk_df, x='sex', y='target', title="性别与患病率", color='target', color_continuous_scale='Blues')
            fig.update_layout(template='simple_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif analysis_type == "胸痛类型患病率":
            cp_risk_df = chest_pain_risk(df)
            fig = px.bar(cp_risk_df, x='cp', y='target', title="胸痛类型与患病率", color='target', color_continuous_scale='Blues')
            fig.update_layout(template='simple_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 选择特征与患病率的关系可视化")
            col_x = st.selectbox("选择特征 (X轴)", options=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], index=0)
            chart_type = st.radio("图表类型", ["箱线图 (Boxplot)", "直方图 (Histogram)", "散点图 + 抖动"])

            df_plot = df.copy()
            df_plot['target_str'] = df_plot['target'].map({0: '健康', 1: '患病'})

            if col_x not in df_plot.columns:
                st.error(f"数据中未找到列: {col_x}")
            else:
                # 转换数值
                if df_plot[col_x].dtype == 'object':
                    df_plot[col_x] = df_plot[col_x].astype(str).str.strip()
                df_plot[col_x] = pd.to_numeric(df_plot[col_x], errors='coerce')
                valid_count = df_plot[col_x].notna().sum()

                if valid_count == 0:
                    st.error(f"❌ 特征列 '{col_x}' 无法转换为数值（有效数据为0），无法绘图。")
                    st.write("原始列部分值:", df[col_x].dropna().head(10).tolist())
                else:
                    st.info(f"✅ 有效数据量: {valid_count} / {len(df_plot)}（已自动将非数值转为NaN）")
                    if valid_count / len(df_plot) < 0.1:
                        st.warning(f"有效数据占比很低 ({valid_count/len(df_plot):.1%})，图表可能稀疏。")

                    try:
                        if chart_type == "箱线图 (Boxplot)":
                            fig = px.box(
                                df_plot,
                                x='target_str',
                                y=col_x,
                                color='target_str',
                                title=f"{col_x} 按患病与否的分布",
                                labels={'target_str': '患病 (健康/患病)', col_x: col_x},
                                color_discrete_map={'健康': '#3182ce', '患病': '#e53e3e'}
                            )
                        elif chart_type == "直方图 (Histogram)":
                            fig = px.histogram(
                                df_plot,
                                x=col_x,
                                color='target_str',
                                nbins=30,
                                title=f"{col_x} 分布的患病对比",
                                labels={col_x: col_x, 'count': '频数'},
                                barmode='overlay',
                                opacity=0.6,
                                color_discrete_map={'健康': '#3182ce', '患病': '#e53e3e'}
                            )
                        else:
                            np.random.seed(42)
                            df_plot['target_jitter'] = df_plot['target'] + np.random.uniform(-0.08, 0.08, size=len(df_plot))
                            fig = px.scatter(
                                df_plot,
                                x=col_x,
                                y='target_jitter',
                                color='target_str',
                                title=f"{col_x} 与患病关系 (抖动散点图)",
                                labels={col_x: col_x, 'target_jitter': '患病 (抖动值)'},
                                opacity=0.5,
                                color_discrete_map={'健康': '#3182ce', '患病': '#e53e3e'}
                            )
                            fig.update_layout(
                                yaxis=dict(tickvals=[0, 1], ticktext=['健康 (0)', '患病 (1)'])
                            )
                        # 强制设置高度，确保可见
                        fig.update_layout(template='simple_white', height=500)

                        # 检查是否有数据轨迹
                        if len(fig.data) == 0:
                            st.warning("生成的图表没有任何数据轨迹，可能是由于数据过滤或数值问题。")
                        else:
                            # 根据用户选择使用渲染方式
                            if use_html_fallback:
                                # 使用 HTML 组件渲染（绕过 Streamlit 的渲染问题）
                                html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
                                components.html(html_str, height=550)
                            else:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"绘图出错: {e}")
                        st.exception(e)

        with st.expander("查看原始数据样本"):
            st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.subheader("基于 Apriori 的指标准则发现")
        st.markdown("发现各项生理指标与患病之间的强关联规则")
        with st.expander("算法参数设置", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                min_support = st.slider("最小支持度", 0.01, 0.5, 0.05, 0.01)
            with col_b:
                min_confidence = st.slider("最小置信度", 0.1, 1.0, 0.7, 0.05)
        if st.button("执行关联规则挖掘", use_container_width=True):
            with st.spinner("正在发掘强关联规则..."):
                try:
                    rules_df = run_apriori_rules(df, min_support, min_confidence)
                    if rules_df.empty:
                        st.warning("未找到符合条件的关联规则，请尝试降低支持度或置信度阈值。")
                    else:
                        st.success(f"挖掘完成！找到 {len(rules_df)} 条规则。")
                        st.dataframe(rules_df, use_container_width=True)
                        csv = rules_df.to_csv(index=False).encode('utf-8')
                        st.download_button("导出规则为 CSV", data=csv, file_name="rules.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"挖掘出错: {e}")

    with tab3:
        st.subheader("患者风险智能预估")
        st.markdown("输入患者当期检测指标，利用随机森林分类器预测心血管疾病风险。")
        with st.spinner("正在训练预测模型..."):
            model, scaler, feature_cols = train_model(df)
        with st.form("risk_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("年龄", 1, 120, 50)
                sex = st.selectbox("性别", [1,0], format_func=lambda x: "男" if x else "女")
                cp = st.selectbox("胸痛类型", [0,1,2,3], format_func=lambda x: ["典型心绞痛","非典型心绞痛","非心绞痛","无症状"][x])
                trestbps = st.number_input("静息血压", 80, 250, 120)
            with col2:
                chol = st.number_input("胆固醇", 100, 600, 200)
                fbs = st.selectbox("空腹血糖>120", [0,1], format_func=lambda x: "是" if x else "否")
                restecg = st.selectbox("静息心电图", [0,1,2])
                thalach = st.number_input("最大心率", 60, 220, 150)
            with col3:
                exang = st.selectbox("运动诱发心绞痛", [0,1], format_func=lambda x: "是" if x else "否")
                oldpeak = st.number_input("ST段压低", 0.0, 10.0, 1.0, 0.1)
                slope = st.selectbox("ST段斜率", [0,1,2])
                ca = st.selectbox("主要血管数", [0,1,2,3,4])
                thal = st.selectbox("地中海贫血", [0,1,2,3])
            submitted = st.form_submit_button("开始智能预测", use_container_width=True)
            if submitted:
                input_arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
                input_scaled = scaler.transform(input_arr)
                prob = model.predict_proba(input_scaled)[0][1]
                st.metric("预测患病风险", f"{prob:.1%}")
                st.progress(prob)
                if prob > 0.5:
                    st.error(f"⚠️ 高风险！患病概率 {prob:.1%}")
                    st.toast("请注意防范！", icon="⚠️")
                else:
                    st.success(f"✅ 低风险，患病概率 {prob:.1%}")
                    st.balloons()

if __name__ == "__main__":
    main()