import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.tracking
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'XGBoost': 'models/xgboost_model.pkl',
        'LightGBM': 'models/lightgbm_model.pkl',
        'Random Forest': 'models/randomforest_model.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
    
    return models

@st.cache_data
def load_data_info():
    """Load data information"""
    if os.path.exists('models/data_info.json'):
        with open('models/data_info.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_comparison():
    """Load model comparison results"""
    if os.path.exists('models/model_comparison.csv'):
        return pd.read_csv('models/model_comparison.csv')
    return None

def create_input_form(data_info):
    """Create input form for prediction"""
    st.markdown('<p class="sub-header">🏠 House Features</p>', unsafe_allow_html=True)
    
    inputs = {}
    feature_ranges = data_info['feature_ranges']
    
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Area'] = st.number_input(
            "Area (sq ft)", 
            min_value=int(feature_ranges['Area']['min']), 
            max_value=int(feature_ranges['Area']['max']),
            value=int((feature_ranges['Area']['min'] + feature_ranges['Area']['max']) / 2),
            key="input_area"
        )
        
        inputs['Bedrooms'] = st.number_input(
            "Bedrooms", 
            min_value=int(feature_ranges['Bedrooms']['min']), 
            max_value=int(feature_ranges['Bedrooms']['max']),
            value=int((feature_ranges['Bedrooms']['min'] + feature_ranges['Bedrooms']['max']) / 2),
            key="input_bedrooms"
        )
        
        inputs['Bathrooms'] = st.number_input(
            "Bathrooms", 
            min_value=int(feature_ranges['Bathrooms']['min']), 
            max_value=int(feature_ranges['Bathrooms']['max']),
            value=int((feature_ranges['Bathrooms']['min'] + feature_ranges['Bathrooms']['max']) / 2),
            key="input_bathrooms"
        )
        
        inputs['Age'] = st.number_input(
            "Age (years)", 
            min_value=int(feature_ranges['Age']['min']), 
            max_value=int(feature_ranges['Age']['max']),
            value=int((feature_ranges['Age']['min'] + feature_ranges['Age']['max']) / 2),
            key="input_age"
        )
    
    with col2:
        inputs['PricePerSqFt'] = st.number_input(
            "Price Per Sq Ft", 
            min_value=feature_ranges['PricePerSqFt']['min'], 
            max_value=feature_ranges['PricePerSqFt']['max'],
            value=(feature_ranges['PricePerSqFt']['min'] + feature_ranges['PricePerSqFt']['max']) / 2,
            key="input_price_per_sqft"
        )
        
        inputs['TotalRooms'] = st.number_input(
            "Total Rooms", 
            min_value=int(feature_ranges['TotalRooms']['min']), 
            max_value=int(feature_ranges['TotalRooms']['max']),
            value=int((feature_ranges['TotalRooms']['min'] + feature_ranges['TotalRooms']['max']) / 2),
            key="input_total_rooms"
        )
    
    # Handle categorical features (one-hot encoded)
    categorical_features = [f for f in data_info['feature_names'] if f not in inputs.keys()]
    
    # Group categorical features by prefix
    location_features = [f for f in categorical_features if f.startswith('Location_')]
    condition_features = [f for f in categorical_features if f.startswith('Condition_')]
    garage_features = [f for f in categorical_features if f.startswith('Garage_')]
    
    st.markdown('<p class="sub-header">📍 Categorical Features</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if location_features:
            location_options = ['None'] + [f.replace('Location_', '') for f in location_features]
            selected_location = st.selectbox("Location", location_options, key="input_location")
            for feature in location_features:
                inputs[feature] = 1 if feature == f"Location_{selected_location}" else 0
    
    with col2:
        if condition_features:
            condition_options = ['None'] + [f.replace('Condition_', '') for f in condition_features]
            selected_condition = st.selectbox("Condition", condition_options, key="input_condition")
            for feature in condition_features:
                inputs[feature] = 1 if feature == f"Condition_{selected_condition}" else 0
    
    with col3:
        if garage_features:
            garage_options = ['None'] + [f.replace('Garage_', '') for f in garage_features]
            selected_garage = st.selectbox("Garage", garage_options, key="input_garage")
            for feature in garage_features:
                inputs[feature] = 1 if feature == f"Garage_{selected_garage}" else 0
    
    # Ensure all features are present
    for feature in data_info['feature_names']:
        if feature not in inputs:
            inputs[feature] = 0
    
    return inputs

def make_prediction(model, inputs, feature_names):
    """Make prediction using selected model"""
    # Create input array in correct order
    input_array = np.array([[inputs[feature] for feature in feature_names]])
    prediction = model.predict(input_array)[0]
    return prediction

def process_batch_file(uploaded_file, models, selected_model, feature_names):
    """Process batch prediction file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            st.error(f"Missing features in uploaded file: {missing_features}")
            return None
        
        # Make predictions
        model = models[selected_model]
        predictions = model.predict(df[feature_names])
        
        # Add predictions to dataframe
        df['Predicted_Price'] = predictions
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def show_model_comparison():
    """Display model comparison dashboard"""
    st.markdown('<p class="sub-header">📊 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    comparison_df = load_model_comparison()
    if comparison_df is None:
        st.error("Model comparison data not found!")
        return
    
    # Display metrics table
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE comparison
        fig_rmse = px.bar(
            comparison_df, 
            x='Model', 
            y='RMSE',
            title='Root Mean Square Error (Lower is Better)',
            color='RMSE',
            color_continuous_scale='Reds_r'
        )
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        # MAE comparison
        fig_mae = px.bar(
            comparison_df, 
            x='Model', 
            y='MAE',
            title='Mean Absolute Error (Lower is Better)',
            color='MAE',
            color_continuous_scale='Oranges_r'
        )
        fig_mae.update_layout(showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        # R² comparison
        fig_r2 = px.bar(
            comparison_df, 
            x='Model', 
            y='R²',
            title='R² Score (Higher is Better)',
            color='R²',
            color_continuous_scale='Greens'
        )
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # MSE comparison
        fig_mse = px.bar(
            comparison_df, 
            x='Model', 
            y='MSE',
            title='Mean Square Error (Lower is Better)',
            color='MSE',
            color_continuous_scale='Blues_r'
        )
        fig_mse.update_layout(showlegend=False)
        st.plotly_chart(fig_mse, use_container_width=True)
    
    # Radar chart for overall comparison
    st.markdown('<p class="sub-header">🎯 Overall Model Performance</p>', unsafe_allow_html=True)
    
    # Normalize metrics for radar chart (0-1 scale)
    normalized_df = comparison_df.copy()
    normalized_df['RMSE_norm'] = 1 - (normalized_df['RMSE'] - normalized_df['RMSE'].min()) / (normalized_df['RMSE'].max() - normalized_df['RMSE'].min())
    normalized_df['MAE_norm'] = 1 - (normalized_df['MAE'] - normalized_df['MAE'].min()) / (normalized_df['MAE'].max() - normalized_df['MAE'].min())
    normalized_df['R²_norm'] = normalized_df['R²']
    normalized_df['MSE_norm'] = 1 - (normalized_df['MSE'] - normalized_df['MSE'].min()) / (normalized_df['MSE'].max() - normalized_df['MSE'].min())
    
    fig_radar = go.Figure()
    
    for _, row in normalized_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['RMSE_norm'], row['MAE_norm'], row['R²_norm'], row['MSE_norm']],
            theta=['RMSE', 'MAE', 'R²', 'MSE'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Normalized Model Performance (Higher is Better)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Best model recommendation
    best_model_idx = comparison_df['RMSE'].idxmin()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    best_rmse = comparison_df.loc[best_model_idx, 'RMSE']
    best_r2 = comparison_df.loc[best_model_idx, 'R²']
    
    st.success(f"🏆 **Recommended Model: {best_model}** (RMSE: {best_rmse:.2f}, R²: {best_r2:.4f})")

def show_mlflow_info():
    """Display MLflow information"""
    st.markdown('<p class="sub-header">🔬 MLflow Experiment Tracking</p>', unsafe_allow_html=True)
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get experiment info
        experiment = mlflow.get_experiment_by_name("house-price-automl")
        if experiment:
            st.info(f"**Experiment:** {experiment.name}")
            st.info(f"**Experiment ID:** {experiment.experiment_id}")
            st.info(f"**Tracking URI:** {mlflow.get_tracking_uri()}")
            
            # Get runs
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs:
                st.success(f"**Total Runs:** {len(runs)}")
                
                # Display run information
                run_data = []
                for run in runs:
                    run_data.append({
                        'Run Name': run.data.tags.get('mlflow.runName', 'Unknown'),
                        'Run ID': run.info.run_id[:8] + '...',
                        'Status': run.info.status,
                        'RMSE': run.data.metrics.get('rmse', 'N/A'),
                        'R²': run.data.metrics.get('r2', 'N/A'),
                        'Start Time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                run_df = pd.DataFrame(run_data)
                st.dataframe(run_df, use_container_width=True)
                
                st.markdown("**To view detailed MLflow UI:**")
                st.code("mlflow ui --backend-store-uri file:./mlruns", language="bash")
            else:
                st.warning("No runs found in the experiment")
        else:
            st.error("MLflow experiment 'house-price-automl' not found")
    except Exception as e:
        st.error(f"Error accessing MLflow: {str(e)}")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🏠 House Price Prediction App</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🎯 Single Prediction", "📊 Batch Prediction", "📈 Model Comparison", "🔬 MLflow Tracking"],
        key="sidebar_page_radio"
    )
    
    # Load data
    models = load_models()
    data_info = load_data_info()
    
    if not models:
        st.error("❌ No models found! Please run the notebook first to train models.")
        return
    
    if not data_info:
        st.error("❌ Data info not found! Please run the notebook first.")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Available Models:**")
    for model_name in models.keys():
        st.sidebar.write(f"✅ {model_name}")
    
    # Single Prediction Page
    if page == "🎯 Single Prediction":
        st.markdown("### Make a Single House Price Prediction")
        
        # Model selection
        selected_model = st.selectbox("Select Model:", list(models.keys()), key="single_pred_model_select")
        
        # Input form
        inputs = create_input_form(data_info)
        
        # Prediction button
        if st.button("🔮 Predict Price", type="primary", key="single_predict_button"):
            try:
                prediction = make_prediction(models[selected_model], inputs, data_info['feature_names'])
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Selected Model", selected_model)
                with col2:
                    st.metric("Predicted Price", f"${prediction:,.2f}")
                with col3:
                    st.metric("Price Range", f"${prediction*0.9:,.0f} - ${prediction*1.1:,.0f}")
                
                # Show input summary
                st.markdown('<p class="sub-header">📋 Input Summary</p>', unsafe_allow_html=True)
                input_df = pd.DataFrame([inputs]).T
                input_df.columns = ['Value']
                input_df = input_df[input_df['Value'] != 0]  # Show only non-zero values
                st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Batch Prediction Page
    elif page == "📊 Batch Prediction":
        st.markdown("### Batch House Price Predictions")
        
        # Model selection
        selected_model = st.selectbox("Select Model:", list(models.keys()), key="batch_pred_model_select")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with house features",
            type=['csv'],
            help="CSV file should contain all required features",
            key="batch_file_uploader"
        )
        
        if uploaded_file is not None:
            # Show required features
            with st.expander("📋 Required Features in CSV"):
                st.write("Your CSV file must contain these columns:")
                feature_df = pd.DataFrame({'Required Features': data_info['feature_names']})
                st.dataframe(feature_df, use_container_width=True)
            
            # Process file
            if st.button("🚀 Process Batch Predictions", type="primary", key="batch_predict_button"):
                with st.spinner("Processing predictions..."):
                    results_df = process_batch_file(uploaded_file, models, selected_model, data_info['feature_names'])
                
                if results_df is not None:
                    st.success(f"✅ Predictions completed for {len(results_df)} houses!")
                    
                    # Display results
                    st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Houses", len(results_df))
                    with col2:
                        st.metric("Avg Predicted Price", f"${results_df['Predicted_Price'].mean():,.2f}")
                    with col3:
                        st.metric("Min Price", f"${results_df['Predicted_Price'].min():,.2f}")
                    with col4:
                        st.metric("Max Price", f"${results_df['Predicted_Price'].max():,.2f}")
                    
                    # Price distribution chart
                    fig_hist = px.histogram(
                        results_df, 
                        x='Predicted_Price', 
                        title='Distribution of Predicted Prices',
                        nbins=20
                    )
                    fig_hist.update_xaxis(title="Predicted Price ($)")
                    fig_hist.update_yaxis(title="Count")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Download results
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"house_price_predictions_{selected_model.lower()}.csv",
                        mime="text/csv"
                    )
        else:
            # Show sample format
            st.info("📝 Upload a CSV file to get started!")
            with st.expander("📄 Sample CSV Format"):
                sample_data = {feature: [0] for feature in data_info['feature_names']}
                sample_data['Area'] = [2000]
                sample_data['Bedrooms'] = [3]
                sample_data['Bathrooms'] = [2]
                sample_data['Age'] = [10]
                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df, use_container_width=True)
    
    # Model Comparison Page
    elif page == "📈 Model Comparison":
        show_model_comparison()
    
    # MLflow Tracking Page
    elif page == "🔬 MLflow Tracking":
        show_mlflow_info()

if __name__ == "__main__":
    main()
