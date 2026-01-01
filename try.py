import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RetentionPilot AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD ASSETS (Model & Data)
# ==========================================
@st.cache_data
def load_data():
    # Hardcoded path for local testing as requested
    file_path = r"D:\LLM_model_project\customer_churn_data.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"⚠️ File not found at: {file_path}")
        return None

@st.cache_resource
def load_model():
    # Hardcoded path for local testing as requested
    file_path = r"D:\LLM_model_project\churn_model.pkl"
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Model file not found at: {file_path}")
        return None

df = load_data()
model = load_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📉 RetentionPilot")
st.sidebar.write("Intelligent Churn Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["🔮 Prediction Simulator", "📊 Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("Built for B.Tech Capstone Project")

# ==========================================
# 4. PAGE 1: PREDICTION SIMULATOR
# ==========================================
if page == "🔮 Prediction Simulator":
    st.title("🔮 Real-Time Churn Risk Simulator")
    st.markdown("Adjust customer parameters to predict the probability of churn.")
    
    if model:
        st.markdown("### 📝 Customer Profile")
        
        # --- INPUT FORM ---
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 80, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                usage = st.slider("Monthly Usage (Hours)", 0, 200, 50)
                
            with col2:
                transactions = st.number_input("Num Transactions", min_value=0, max_value=100, value=10)
                complaints = st.slider("Customer Complaints (Past 6 Months)", 0, 10, 0)
                sub_type = st.selectbox("Subscription Type", ["Basic", "Gold", "Premium"])
            
            submitted = st.form_submit_button("🚀 Predict Risk")

        # --- PREDICTION LOGIC ---
        if submitted:
            # 1. Preprocess Input
            gender_val = 1 if gender == "Male" else 0
            
            # One-Hot Encoding for Subscription (Manual)
            is_basic = 1 if sub_type == "Basic" else 0
            is_gold = 1 if sub_type == "Gold" else 0
            is_premium = 1 if sub_type == "Premium" else 0
            
            # Create DataFrame with columns matching training data exactly
            input_cols = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 
                          'Complaints', 'SubscriptionType_Basic', 'SubscriptionType_Gold', 'SubscriptionType_Premium']
            
            input_data = pd.DataFrame([[
                age, gender_val, usage, transactions, complaints,
                is_basic, is_gold, is_premium
            ]], columns=input_cols)
            
            # 2. Predict
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                # 3. Display Result
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col_a, col_b = st.columns([1, 2])
                
                if prediction == 1:
                    with col_a:
                        st.error("🚨 HIGH CHURN RISK")
                        st.metric("Churn Probability", f"{probability:.1%}")
                    with col_b:
                        st.warning("⚠️ **Action Recommended:**")
                        st.write("- Offer a 20% discount on next month's bill.")
                        st.write("- Schedule a call with a Senior Support Agent.")
                else:
                    with col_a:
                        st.success("✅ LOW CHURN RISK")
                        st.metric("Churn Probability", f"{probability:.1%}")
                    with col_b:
                        st.info("👍 **Action Recommended:**")
                        st.write("- Candidate for Upselling to Premium Plan.")
                        st.write("- Send 'Thank You' loyalty email.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Please check if feature names match the trained model.")
    else:
        st.warning("Model could not be loaded. Check file path.")

# ==========================================
# 5. PAGE 2: ANALYTICS DASHBOARD
# ==========================================
elif page == "📊 Analytics Dashboard":
    st.title("📊 Executive Analytics Dashboard")
    st.markdown("Insights derived from historical customer data.")
    
    if df is not None:
        # --- KPI METRICS ---
        total_customers = len(df)
        churn_rate = df['Churn'].mean()
        avg_usage = df['MonthlyUsageHours'].mean()
        high_risk_count = len(df[df['Churn'] == 1])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Customers", f"{total_customers:,}")
        kpi2.metric("Overall Churn Rate", f"{churn_rate:.1%}", "-High" if churn_rate > 0.2 else "Normal")
        kpi3.metric("Avg Usage Hours", f"{avg_usage:.0f} hrs")
        kpi4.metric("At-Risk Customers", f"{high_risk_count}", "Action Needed")
        
        st.markdown("---")

        # --- ROW 1: CHURN DISTRIBUTION & COMPLAINTS ---
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("Churn Distribution")
            fig_pie = px.pie(df, names='Churn', title='Churn vs Retained', 
                             color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with row1_col2:
            st.subheader("Impact of Complaints")
            if 'Complaints' in df.columns:
                fig_box = px.box(df, x='Churn', y='Complaints', color='Churn',
                                 title="Do Churners Complain More?",
                                 color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Complaints data not available for plotting.")

        # --- ROW 2: USAGE PATTERNS ---
        st.subheader("Usage Behavior Analysis")
        if 'MonthlyUsageHours' in df.columns:
            fig_hist = px.histogram(df, x="MonthlyUsageHours", color="Churn", 
                                    title="Usage Hours Distribution (Churn vs Retained)",
                                    barmode='overlay', opacity=0.7,
                                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            st.plotly_chart(fig_hist, use_container_width=True)

        # --- ROW 3: CORRELATION HEATMAP ---
        st.subheader("Feature Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.warning("Dataset could not be loaded. Check file path.")