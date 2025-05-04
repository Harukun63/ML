# app.py - Game of Thrones Themed ML Platform
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="⚔️ Throne of ML - By Samad Kiani",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Game of Thrones Theme
st.markdown("""
<style>
    /* Main App Background */
    body, .stApp {
        background-color: #0a0a0a;
        background-image: url('https://static.hbo.com/game-of-thrones-1-1920x1080.jpg');
        background-size: cover;
        background-attachment: fixed;
        color: #d4af37;
        font-family: 'Game of Thrones', 'Times New Roman', serif;
    }
    
    /* Import Game of Thrones font */
    @import url('https://fonts.cdnfonts.com/css/game-of-thrones');
    
    /* Main Content Container */
    .main {
        background-color: rgba(10, 10, 10, 0.85);
        padding: 2rem;
        border-radius: 5px;
        border: 2px solid #d4af37;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #d4af37 !important;
        font-family: 'Game of Thrones', cursive;
        text-shadow: 2px 2px 3px #000000;
        letter-spacing: 1px;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: rgba(10, 10, 10, 0.9) !important;
        border-right: 2px solid #d4af37;
        font-family: 'Game of Thrones', cursive;
        background-image: url('https://i.imgur.com/5zJQY9a.png');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: bottom right;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8b0000;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #d4af37;
        padding: 0.5rem 1.5rem;
        font-family: 'Game of Thrones', cursive;
        font-size: 14px;
        box-shadow: 0 4px 0 #5a0000;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #a52a2a;
        transform: translateY(2px);
        box-shadow: 0 2px 0 #5a0000;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background-color: #1e3a8a;
        box-shadow: 0 4px 0 #0a1f4d;
    }
    
    .stDownloadButton>button:hover {
        background-color: #2a4a9a;
        box-shadow: 0 2px 0 #0a1f4d;
    }
    
    /* File Uploader */
    .stFileUploader>div>div {
        background-color: rgba(20, 20, 20, 0.9);
        border: 2px dashed #d4af37;
        border-radius: 5px;
    }
    
    /* Expanders */
    .st-expander {
        background-color: rgba(20, 20, 20, 0.9);
        border: 2px solid #d4af37;
        border-radius: 5px;
    }
    
    .st-expanderHeader {
        color: #d4af37 !important;
        font-family: 'Game of Thrones', cursive;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: rgba(20, 20, 20, 0.9) !important;
        color: #d4af37 !important;
        border: 2px solid #d4af37 !important;
    }
    
    /* Input Widgets */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div {
        background-color: rgba(20, 20, 20, 0.9);
        color: #d4af37;
        border: 2px solid #d4af37;
        border-radius: 5px;
        font-family: 'Game of Thrones', cursive;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #d4af37;
    }
    
    /* Tabs */
    .stTabs>div>div>div>button {
        background-color: transparent;
        color: #d4af37;
        border: none;
        font-family: 'Game of Thrones', cursive;
    }
    
    .stTabs>div>div>div>button[aria-selected="true"] {
        color: #ffffff;
        background-color: #8b0000;
        border-bottom: 2px solid #d4af37;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div>div {
        background-color: #d4af37;
    }
    
    /* Alerts */
    .stAlert .st-at {
        background-color: rgba(20, 20, 20, 0.9) !important;
        border: 2px solid #d4af37 !important;
        color: #d4af37 !important;
    }
    
    /* Metric Cards */
    .stMetric {
        background-color: rgba(20, 20, 20, 0.9);
        border: 2px solid #d4af37;
        border-radius: 5px;
        padding: 15px;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: rgba(20, 20, 20, 0.9) !important;
    }
    
    /* House Sigil Decoration */
    .house-sigil {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
        width: 120px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Add House Sigil decoration
    st.markdown("""
    <img src="https://i.imgur.com/5zJQY9a.png" class="house-sigil">
    """, unsafe_allow_html=True)
    
    st.title("⚔️ Throne of ML")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚔️ War Council")
        uploaded_file = st.file_uploader("Upload Battle Data:", type=["csv", "xlsx"])
        
        st.markdown("---")
        st.header("🏹 Strategy Selection")
        model_type = st.selectbox("Choose Your Weapon:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reset the Realm", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header("1. Gather Intelligence")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("⚔️ You need at least 2 numeric battle metrics!")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"🏰 Successfully gathered {len(df)} battle records!")
            
            st.write("### Battle Data Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🗡️ Select Battle Metrics"):
                all_cols = df.columns.tolist()
                target = st.selectbox("Select Your Target:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("Choose Battle Factors:", numeric_cols, default=default_features)
                
                if st.button("Prepare for War"):
                    if len(features) < 1:
                        st.error("⚔️ You need at least one battle factor!")
                    elif target in features:
                        st.error("⚔️ Target cannot be a battle factor!")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        st.success("⚔️ Battle plan ready!")
            
        except Exception as e:
            st.error(f"🔥 Battle error: {str(e)}")
    else:
        st.markdown("""
        ### How to Command Your Forces:
        1. 🏰 Upload battle data (CSV or Excel)  
        2. 🎯 Select your target (what you want to predict)  
        3. ⚔️ Choose battle factors (variables for prediction)  
        4. 🏹 The system will devise the best strategy  
        """)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.header("2. Analyze Battle Patterns")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Battle Factor vs Target")
            selected_feature = st.selectbox("Select factor to analyze:", features)
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
            fig.update_layout({
                'plot_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'paper_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'font': {'color': '#d4af37'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### War Council Matrix")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='thermal', aspect="auto")
            fig.update_layout({
                'plot_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'paper_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'font': {'color': '#d4af37'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("⚔️ Train Your Army"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Train Your Forces")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(f"⚔️ Training {model_type} forces..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            st.success("🏆 Victory! Training complete!")
            st.balloons()

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Battle Results")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Battle Error", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}", help="Lower is better")
        with col2:
            st.metric("War Power", f"{r2_score(y_test, y_pred):.2f}", help="1.0 is perfect")
        
        st.write("### Actual vs Predicted Outcomes")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(color='#8b0000', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(color='#d4af37', size=10)
        ))
        fig.update_layout(
            xaxis_title="Battle Index", 
            yaxis_title="Outcome", 
            height=500,
            plot_bgcolor='rgba(20, 20, 20, 0.9)',
            paper_bgcolor='rgba(20, 20, 20, 0.9)',
            font=dict(color='#d4af37')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write("### Battle Factor Importance")
            importance = pd.DataFrame({'Factor': st.session_state.features, 'Power': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Power', ascending=False)
            fig = px.bar(
                importance, 
                x='Power', 
                y='Factor', 
                orientation='h', 
                color='Power', 
                color_continuous_scale='thermal'
            )
            fig.update_layout({
                'plot_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'paper_bgcolor': 'rgba(20, 20, 20, 0.9)',
                'font': {'color': '#d4af37'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📜 Download Battle Report", csv, "battle_results.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
