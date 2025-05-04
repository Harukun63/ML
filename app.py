# app.py - Universal ML Platform with Anime Theme
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
    page_title="Anime ML - By Samad Kiani",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Anime Theme
st.markdown("""
<style>
    /* Main App Background */
    body, .stApp {
        background-color: #ffecf5;
        background-image: url('https://i.imgur.com/X6QX0y3.png');
        background-size: cover;
        background-attachment: fixed;
        color: #5a3d5a;
        font-family: 'Anime Ace', sans-serif;
    }
    
    /* Import Anime-style font */
    @import url('https://fonts.cdnfonts.com/css/anime-ace');
    
    /* Main Content Container */
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #ff85a2;
        box-shadow: 0 0 15px rgba(255, 105, 180, 0.3);
        backdrop-filter: blur(5px);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ff2d7a !important;
        font-family: 'Anime Ace', cursive;
        text-shadow: 2px 2px 0px rgba(255, 255, 255, 0.5);
        letter-spacing: 1px;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: rgba(255, 236, 245, 0.9) !important;
        border-right: 3px solid #ff85a2;
        font-family: 'Anime Ace', cursive;
        background-image: url('https://i.imgur.com/JYw7QxX.png');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: bottom right;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff85a2;
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-family: 'Anime Ace', cursive;
        font-size: 14px;
        box-shadow: 0 4px 0 #d45d7d;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        background-color: #ff6b8b;
        transform: translateY(2px);
        box-shadow: 0 2px 0 #d45d7d;
    }
    
    .stButton>button:active {
        transform: translateY(4px);
        box-shadow: none;
    }
    
    /* Sparkle effect */
    .stButton>button:after {
        content: '★';
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover:after {
        top: 5px;
        opacity: 1;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background-color: #8a6dff;
        box-shadow: 0 4px 0 #6a4dcc;
    }
    
    .stDownloadButton>button:hover {
        background-color: #7a5dff;
        box-shadow: 0 2px 0 #6a4dcc;
    }
    
    /* File Uploader */
    .stFileUploader>div>div {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px dashed #ff85a2;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader>div>div:hover {
        background-color: rgba(255, 255, 255, 0.95);
        border-color: #ff2d7a;
    }
    
    /* Expanders */
    .st-expander {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #ff85a2;
        border-radius: 15px;
    }
    
    .st-expanderHeader {
        color: #ff2d7a !important;
        font-family: 'Anime Ace', cursive;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #5a3d5a !important;
        border: 2px solid #ff85a2 !important;
        border-radius: 10px !important;
    }
    
    /* Input Widgets */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div {
        background-color: rgba(255, 255, 255, 0.9);
        color: #5a3d5a;
        border: 2px solid #ff85a2;
        border-radius: 15px;
        font-family: 'Anime Ace', cursive;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #ff85a2;
    }
    
    /* Tabs */
    .stTabs>div>div>div>button {
        background-color: transparent;
        color: #5a3d5a;
        border: none;
        font-family: 'Anime Ace', cursive;
        transition: all 0.3s ease;
    }
    
    .stTabs>div>div>div>button[aria-selected="true"] {
        color: #ff2d7a;
        transform: scale(1.05);
        text-decoration: underline;
        text-underline-offset: 5px;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div>div {
        background-color: #ff85a2;
        border-radius: 10px;
    }
    
    /* Alerts */
    .stAlert .st-at {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #ff85a2 !important;
        color: #5a3d5a !important;
        border-radius: 15px;
    }
    
    /* Metric Cards */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #ff85a2;
        border-radius: 15px;
        padding: 15px;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px;
    }
    
    /* Special anime decorations */
    .anime-char {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
        width: 150px;
        transition: all 0.5s ease;
    }
    
    .anime-char:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    /* Sparkle animation */
    @keyframes sparkle {
        0% { transform: scale(0); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: scale(1.5); opacity: 0; }
    }
    
    .sparkle {
        position: absolute;
        width: 10px;
        height: 10px;
        background-color: #fff;
        border-radius: 50%;
        pointer-events: none;
        animation: sparkle 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Add anime character decoration
    st.markdown("""
    <img src="https://i.imgur.com/JYw7QxX.png" class="anime-char">
    """, unsafe_allow_html=True)
    
    st.title("🌸 Anime ML Magic")
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
        st.header("⚡ Magic Settings")
        uploaded_file = st.file_uploader("Upload Your Dataset:", type=["csv", "xlsx"])
        
        st.markdown("---")
        st.header("✨ Spell Selection")
        model_type = st.selectbox("Choose Your Magic:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Power:", 0.1, 0.5, 0.2)
        st.button("Reset Everything", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header("1. Summon Your Data")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("💢 You need at least 2 numeric spells to begin!")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✨ Successfully summoned {len(df)} data spirits!")
            
            st.write("### Data Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🔮 Select Magic Components"):
                all_cols = df.columns.tolist()
                target = st.selectbox("Select Your Target:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("Choose Magic Ingredients:", numeric_cols, default=default_features)
                
                if st.button("Cast Selection Spell"):
                    if len(features) < 1:
                        st.error("💢 You need at least one magic ingredient!")
                    elif target in features:
                        st.error("💢 Target can't be used as ingredient!")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        st.success("🌟 Spell cast successfully!")
            
        except Exception as e:
            st.error(f"💥 Spell failed: {str(e)}")
    else:
        st.markdown("""
        ### How to Use Your Magic:
        1. 🌸 Upload any CSV or Excel file with numeric data  
        2. 🎯 Select your target (what you want to predict)  
        3. ✨ Choose magic ingredients (variables for prediction)  
        4. 🧙‍♀️ The system will automatically cast the spells  
        """)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.header("2. Analyze Magic Patterns")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Ingredient-Target Relationship")
            selected_feature = st.selectbox("Select ingredient to analyze:", features)
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
            fig.update_layout({
                'plot_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'paper_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'font': {'color': '#5a3d5a'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Magic Synergy Matrix")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='magma', aspect="auto")
            fig.update_layout({
                'plot_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'paper_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'font': {'color': '#5a3d5a'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🌀 Begin Magic Training"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Train Your Magic")
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
        
        with st.spinner(f"🌀 Training {model_type} magic..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            st.success("🎉 Magic training complete!")
            st.balloons()

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Magic Results")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Magic Error", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}", help="Lower is better")
        with col2:
            st.metric("Magic Power", f"{r2_score(y_test, y_pred):.2f}", help="1.0 is perfect")
        
        st.write("### Actual vs Predicted Magic")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(color='#8a6dff', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(color='#ff85a2', size=10)
        ))
        fig.update_layout(
            xaxis_title="Spell Index", 
            yaxis_title="Magic Power", 
            height=500,
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(color='#5a3d5a')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write("### Ingredient Importance")
            importance = pd.DataFrame({'Ingredient': st.session_state.features, 'Power': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Power', ascending=False)
            fig = px.bar(
                importance, 
                x='Power', 
                y='Ingredient', 
                orientation='h', 
                color='Power', 
                color_continuous_scale='magma'
            )
            fig.update_layout({
                'plot_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'paper_bgcolor': 'rgba(255, 255, 255, 0.9)',
                'font': {'color': '#5a3d5a'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📜 Download Spell Results", csv, "anime_magic_results.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
