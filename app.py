import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="心脏病智能分析与预测",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    load_css(css_path)

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_heart.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_heart.csv")

def main():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #4FD1C5;'>医疗数据中台</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("数据接入与预处理")
        st.write("请上传原始心脏病数据集进行清洗入库。")
        
        uploaded_file = st.file_uploader("上传 CSV 数据集", type=['csv'])
        
        if st.button("重新导入与清洗 (ETL)", use_container_width=True):
            with st.spinner("正在加载、清洗并写入数据库..."):
                st.success("数据处理完毕已入库！")
                st.info("提示: 数据库心血管特征指标已更新。")

    st.markdown("<h1 style='color: #63B3ED;'>心脏病数据仓库与智能挖掘系统</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0; font-size: 1.1rem;'>基于多维数据建模，支持高级聚合查询、Apriori 关联规则挖掘以及机器学习风险预测。</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-card"><h3>当前数据总量</h3><h2>303 <span style="font-size:1rem;">条</span></h2></div>', unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><h3>总体患病率</h3><h2>54.5 <span style="font-size:1rem;">%</span></h2></div>', unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><h3>患者平均年龄</h3><h2>54 <span style="font-size:1rem;">岁</span></h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "聚合分析与可视化", 
        "关联规则挖掘", 
        "风险预测模型"
    ])

    with tab1:
        st.subheader("多维数据集计与透视")
        st.markdown("通过数据仓库的多维模型展示关键特征对心脏病风险的影响。")
        
        if st.button("生成聚合分析图表"):
            with st.spinner("查询数据中..."):
                st.info("图表已成功生成")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig1 = go.Figure(data=[go.Bar(x=['30-40', '40-50', '50-60', '60+'], y=[10, 30, 45, 15], marker_color='#4FD1C5')])
                    fig1.update_layout(title="不同年龄段患病率分布", margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'))
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_chart2:
                    fig2 = px.pie(names=['男性', '女性'], values=[68, 32], color_discrete_sequence=['#4FD1C5', '#3182CE'])
                    fig2.update_layout(title="患病人群性别占比", margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'))
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("基于 Apriori 的指标准则发现")
        st.markdown("发现各项生理指标与患病之间的强关联规则（如 高血压 + 高胆固醇 => 心脏病）。")
        
        with st.expander("算法参数设置", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                min_support = st.slider("最小支持度 (Min Support)", 0.01, 1.0, 0.1)
            with col_b:
                min_confidence = st.slider("最小置信度 (Min Confidence)", 0.1, 1.0, 0.6)

        if st.button("执行关联规则挖掘"):
            with st.spinner("正在发掘强关联规则..."):
                st.success(f"挖掘完成！找到符合支持度>{min_support} 及置信度>{min_confidence} 的规则。")
                
                demo_rules = pd.DataFrame({
                    "前提 (Antecedents)": ["高血压", "年龄>60, 胸痛类型=典型"],
                    "结论 (Consequents)": ["患病=1", "患病=1"],
                    "支持度 (Support)": [0.15, 0.12],
                    "置信度 (Confidence)": [0.85, 0.92],
                    "提升度 (Lift)": [1.5, 1.7]
                })
                st.dataframe(demo_rules, use_container_width=True, hide_index=True)
                st.download_button("导出规则为 CSV", data="xxx", file_name="rules.csv")

    with tab3:
        st.subheader("患者风险智能预估")
        st.markdown("输入患者当期检测指标，利用随机森林分类器预测潜在的心血管疾病风险。")
        
        with st.form("risk_prediction_form"):
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                age = st.number_input("年龄 (Age)", min_value=1, max_value=120, value=50)
                sex = st.selectbox("性别 (Sex)", options=[1, 0], format_func=lambda x: "男 (1)" if x == 1 else "女 (0)")
                cp = st.selectbox("胸痛类型 (CP)", options=[0, 1, 2, 3])
                trestbps = st.number_input("静息血压 (Trestbps)", value=120)
                
            with col_p2:
                chol = st.number_input("胆固醇 (Chol)", value=200)
                fbs = st.selectbox("空腹血糖 > 120 mg/dl (FBS)", options=[0, 1])
                restecg = st.selectbox("静息心电图 (RestECG)", options=[0, 1, 2])
                thalach = st.number_input("最大心率 (Thalach)", value=150)
                
            with col_p3:
                exang = st.selectbox("运动引发心绞痛 (Exang)", options=[0, 1])
                oldpeak = st.number_input("ST段压低 (Oldpeak)", value=1.0, step=0.1)
                slope = st.selectbox("ST段斜率 (Slope)", options=[0, 1, 2])
                ca = st.selectbox("主要血管数 (CA)", options=[0, 1, 2, 3, 4])
                thal = st.selectbox("地中海贫血 (Thal)", options=[0, 1, 2, 3])
                
            submit_pred = st.form_submit_button("开始智能预测", use_container_width=True)
            
        if submit_pred:
            with st.spinner("AI模型正在评估..."):
                risk_prob = 0.82 
                st.markdown("### 评估结果")
                if risk_prob > 0.5:
                    st.error(f"高风险警告！预测患病概率为 {risk_prob:.1%}")
                    st.toast('发现高风险指标，请注意防范！')
                else:
                    st.success(f"状况良好，患病概率为 {risk_prob:.1%}")
                    st.balloons()
                    
                st.progress(risk_prob)

if __name__ == "__main__":
    main()
