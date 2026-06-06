import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="Rebar Analysis Dashboard", layout="wide")
st.title("🏗️ Reinforcement Bars Dashboard")

# Apply your Custom Plotly Theme
color_template = {
    "layout": {
        "width": 600,
        "height": 400,
        "colorway": [
            "#AFA9E9", "#8FDBA8", "#CF98AA", "#7C3C60", "#740580",
            "#18A999", "#5A20CB", "#FFB200", "#06D6A0", "#118AB2",
            "#EF476F", "#FFD166", "#073B4C", "#8338EC", "#FB5607"
        ],
        "plot_bgcolor": "#F8E6F8",   
        "paper_bgcolor": "#F8E6F8",  
        "font": {"family": "Arial", "size": 14, "color": "#333"},
        "xaxis": {
            "showgrid": True, "gridcolor": "#FFFFFF", 
            "showline": True, "linecolor": "#333333", "linewidth": 1
        },
        "yaxis": {
            "showgrid": True, "gridcolor": "#FFFFFF", 
            "showline": True, "linecolor": "#333333", "linewidth": 1
        },
        "title": {"font": {"size": 20, "color": "#740580"}},
    }
}
pio.templates["custom_theme"] = color_template
pio.templates.default = "custom_theme"

# ==========================================
# 2. DATA LOADING & CLEANING (Cached for speed)
# ==========================================
@st.cache_data
def load_data():
    # Load raw data
    df = pd.read_csv('Reinforcement Bars.csv')
    
    # Clean data (Drop grand total row)
    df_clean = df.iloc[:-1].copy()
    
    # Fix numeric columns
    df_clean['Bar Diameter Num'] = df_clean['Bar Diameter'].str.replace(' mm', '').astype(float)
    df_clean['Max Length Num'] = df_clean['Maximum bar length'].str.replace(' mm', '').astype(float)
    
    # Update Workset
    df_clean['Workset'] = df_clean['Workset'].replace({
        'STR-Model': 'STR-Air Cooled Plant', 
        'STR-TRENCH': 'STR-Air Cooled Plant'
    })
    
    return df_clean

df = load_data()
unique_diameters = sorted(df['Bar Diameter Num'].dropna().unique())

# ==========================================
# 3. DASHBOARD LAYOUT & CHARTS
# ==========================================

# --- ROW 1 ---
col1, col2 = st.columns(2)

with col1:
    fig_workset = px.histogram(df, x='Workset', title='Total Rebar Count per Workset', text_auto=True)
    fig_workset.update_layout(bargap=0.2)
    st.plotly_chart(fig_workset, use_container_width=True)

with col2:
    fig_workset_dia = px.histogram(
        df, x='Workset', color='Bar Diameter', barmode='group',
        title='Rebar Count by Workset & Diameter', text_auto=True
    )
    fig_workset_dia.update_layout(bargap=0.2)
    st.plotly_chart(fig_workset_dia, use_container_width=True)

# --- ROW 2 ---
col3, col4 = st.columns(2)

with col3:
    fig_host_workset = px.histogram(
        df, x='Host Category', color='Workset', barmode='group',
        title='Rebar Count by Host Category', text_auto=True
    )
    fig_host_workset.update_layout(bargap=0.2)
    st.plotly_chart(fig_host_workset, use_container_width=True)

with col4:
    fig_weight = px.histogram(
        df, x='Bar Diameter Num', y='WEIGHT IN KG', color='Workset',
        histfunc='sum', barmode='group', title='Total Weight (kg) by Bar Diameter',
        text_auto='.0f'
    )
    fig_weight.update_layout(bargap=0.2)
    fig_weight.update_xaxes(tickvals=unique_diameters)
    st.plotly_chart(fig_weight, use_container_width=True)

# --- ROW 3 ---
st.divider() # Adds a nice horizontal line separator

col5, col6 = st.columns(2)

with col5:
    fig_scatter = px.scatter(
        df, x='Bar Diameter Num', y='Max Length Num', color='Host Category',
        title='Maximum Bar Length vs. Bar Diameter', opacity=0.7
    )
    fig_scatter.update_xaxes(tickvals=unique_diameters)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col6:
    fig_style = px.histogram(
        df, x='Style', y='Quantity', histfunc='sum',
        title='Total Rebar Quantity by Style', text_auto=True
    )
    fig_style.update_layout(bargap=0.2)
    st.plotly_chart(fig_style, use_container_width=True)

    # ==========================================
    # 4. MACHINE LEARNING PREDICTOR
    # ==========================================
    st.divider()
    st.header("🔮 Structural Column Steel Estimator")
    st.write(
        "Enter the column dimensions and required bar diameter to predict the total steel weight based on historical project data.")


    # 1. Load the model safely
    @st.cache_resource
    def load_model():
        # This ensures the model is only loaded once, keeping the app incredibly fast
        return joblib.load('knn_rebar_model.pkl')


    model = load_model()

    # 2. Create the User Interface Form
    with st.form("prediction_form"):
        col_input1, col_input2 = st.columns(2)

        with col_input1:
            # Using number inputs for custom dimensions
            length = st.number_input("Column Length (mm)", min_value=100, max_value=20000, value=900)
            width = st.number_input("Column Width (mm)", min_value=100, max_value=20000, value=900)

        with col_input2:
            height = st.number_input("Column Height (mm)", min_value=100, max_value=30000, value=10500)
            # Using a dropdown for diameter to restrict inputs to standard rebar sizes
            diameter = st.selectbox("Bar Diameter (mm)", [8, 10, 12, 14, 16, 18, 20, 22, 25, 32], index=4)

        # The submit button
        submit_button = st.form_submit_button("Predict Total Weight")

    # 3. Calculate and display the result when the button is clicked
    if submit_button:
        # Package the user's inputs into a dataframe EXACTLY as the model was trained on
        input_data = pd.DataFrame({
            'Column_Length': [length],
            'Column_Width': [width],
            'Column_Height': [height],
            'Bar Diameter Num': [diameter]
        })

        # Run the prediction
        predicted_weight = model.predict(input_data)[0]

        # Display the result in a massive green success box!
        st.success(f"### Estimated Total Steel Weight: {predicted_weight:,.2f} KG")
        st.info("📊 **Model Specs:** Powered by K-Nearest Neighbors (R²: 0.74 | MAE: ~520 KG)")