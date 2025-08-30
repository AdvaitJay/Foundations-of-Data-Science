import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from processor import process_data
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import linregress
from content import (
    DASHBOARD_INFO, ANALYSIS_INFO, PREDICTION_METHODOLOGY, 
    DISASTER_INFO, EMERGENCY_RESOURCES, RISK_MESSAGES,
    INTERFACE_LABELS, WARNING_MESSAGES, DISASTER_GUIDE
)

# Configure page
st.set_page_config(
    page_title="Climate Change Impact Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff7f0e;
        padding-left: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .disaster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .safety-tip {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .warning-tip {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .emergency-tip {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("realistic_climate_change_impacts.csv")
    df = process_data(df)
    return df

df = load_data()

# Navigation menu
navigation_menu = [
    "Home",
    "Univariate Analysis", 
    "Bivariate Analysis",
    "Multivariate Analysis",
    "Prediction",
    "Disaster Types & Safety"
]

# Sidebar navigation
st.sidebar.markdown("## Navigation")
user_menu = st.sidebar.radio('Select Analysis Type:', navigation_menu)

# HOME PAGE
if user_menu == 'Home':
    st.markdown(f'<div class="main-header">{DASHBOARD_INFO["title"]}</div>', unsafe_allow_html=True)
    
    # Key Statistics Row - STATISTICAL CALCULATIONS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Countries Analyzed", df['Country'].nunique())
    with col3:
        total_economic_impact = df['EconomicImpact_USD'].sum()
        st.metric("Total Economic Impact", f"${total_economic_impact/1e9:.1f}B")
    with col4:
        total_population_affected = df['PopulationAffected'].sum()
        st.metric("Total Population Affected", f"{total_population_affected/1e6:.1f}M")
    
    st.markdown("---")
    
    # About Section - TEXTUAL FROM CONTENT.PY
    st.markdown('<div class="section-header">About This Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    <p><strong>{DASHBOARD_INFO["title"]}</strong> {DASHBOARD_INFO["description"]}</p>
    """, unsafe_allow_html=True)
    
    for objective in DASHBOARD_INFO["objectives"]:
        st.markdown(f"<li><strong>{objective}</strong></li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key Metrics Analyzed:**")
        for metric in DASHBOARD_INFO["dataset_metrics"]:
            st.markdown(f"- **{metric}**")
    
    with col2:
        # Most affected countries - STATISTICAL CALCULATION
        top_countries = df.groupby('Country')['PopulationAffected'].sum().sort_values(ascending=False).head(5)
        st.markdown("**Most Affected Countries:**")
        for country, population in top_countries.items():
            st.write(f"• {country}: {population/1e6:.1f}M people affected")
    

    # Recent Trends - STATISTICAL CALCULATION
    st.markdown('<div class="section-header">Recent Climate Data (2020-2023)</div>', unsafe_allow_html=True)
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    recent_data = df[df['Year'] >= 2020]
    st.dataframe(recent_data.head(10), use_container_width=True)
    
    # Navigation Guide - TEXTUAL FROM CONTENT.PY
    st.markdown('<div class="section-header">Navigation Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <h4>Explore Different Sections:</h4>
    <ul>
    """, unsafe_allow_html=True)
    
    for section in DASHBOARD_INFO["navigation_sections"]:
        st.markdown(f"<li><strong>{section}</strong></li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)

# UNIVARIATE ANALYSIS
elif user_menu == 'Univariate Analysis':
    st.markdown(f'<div class="main-header">{ANALYSIS_INFO["univariate"]["title"]}</div>', unsafe_allow_html=True)
    st.markdown(ANALYSIS_INFO["univariate"]["description"])
    
    # Temperature Anomaly Analysis - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Temperature Anomaly Distribution</div>', unsafe_allow_html=True)
    st.markdown(f"**Analysis:** {ANALYSIS_INFO['univariate']['temperature_insight']}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['TemperatureAnomaly_C'], bins=40, kde=True, color='steelblue', ax=ax)
    temp_mean = df['TemperatureAnomaly_C'].mean()
    ax.axvline(temp_mean, color='red', linestyle='--', label=f'Mean: {temp_mean:.2f}°C')
    ax.set_title("Distribution of Temperature Anomalies", fontsize=16, fontweight='bold')
    ax.set_xlabel("Temperature Anomaly (°C)")
    ax.legend()
    st.pyplot(fig)
    
    # Temperature statistics
    temp_stats = df['TemperatureAnomaly_C'].describe()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minimum Anomaly", f"{temp_stats['min']:.2f}°C")
    with col2:
        st.metric("Maximum Anomaly", f"{temp_stats['max']:.2f}°C")
    with col3:
        st.metric("Standard Deviation", f"{temp_stats['std']:.2f}°C")
    
    # Disaster Frequency Analysis - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Extreme Weather Event Frequency</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    event_counts = df['ExtremeWeatherEvent'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(event_counts)))
    bars = ax.barh(event_counts.index, event_counts.values, color=colors)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, event_counts.values)):
        percentage = count/len(df)*100
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{count} ({percentage:.1f}%)', 
                ha='left', va='center', fontweight='bold')
    
    ax.set_title("Frequency of Extreme Weather Events", fontsize=16, fontweight='bold')
    ax.set_xlabel("Number of Occurrences")
    plt.tight_layout()
    st.pyplot(fig)
    
    # CO2 Levels Analysis - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Atmospheric CO₂ Levels Distribution</div>', unsafe_allow_html=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    sns.histplot(df['CO2Level_ppm'], bins=40, kde=True, color='coral', ax=ax1)
    ax1.axvline(400, color='red', linestyle='--', label='Critical Level (400 ppm)')
    ax1.set_title("CO₂ Levels Distribution")
    ax1.legend()
    
    # Box plot
    sns.boxplot(x=df['CO2Level_ppm'], color='lightcoral', ax=ax2)
    ax2.set_title("CO₂ Levels Box Plot")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Economic Impact Analysis - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Economic Impact Analysis</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Use log scale for better visualization
    log_economic = np.log10(df['EconomicImpact_USD'] + 1)
    sns.histplot(log_economic, bins=40, kde=True, color='green', alpha=0.7, ax=ax)
    ax.set_title("Economic Impact Distribution (Log Scale)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Log₁₀(Economic Impact + 1)")
    st.pyplot(fig)
    
    # Economic statistics
    econ_stats = df['EconomicImpact_USD'].describe()
    total_impact = df['EconomicImpact_USD'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Impact", f"${total_impact/1e9:.1f}B")
    with col2:
        st.metric("Average Impact", f"${econ_stats['mean']/1e6:.1f}M")
    with col3:
        st.metric("Median Impact", f"${econ_stats['50%']/1e6:.1f}M")
    with col4:
        st.metric("Max Impact", f"${econ_stats['max']/1e6:.1f}M")


# BIVARIATE ANALYSIS
elif user_menu == 'Bivariate Analysis':
    st.markdown(f'<div class="main-header">{ANALYSIS_INFO["bivariate"]["title"]}</div>', unsafe_allow_html=True)
    st.markdown(ANALYSIS_INFO["bivariate"]["description"])
    
    # Temperature vs Economic Impact - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Temperature Anomaly vs Economic Impact</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(
        data=df, x='TemperatureAnomaly_C', y='EconomicImpact_USD',
        hue='ExtremeWeatherEvent', alpha=0.7, s=80, ax=ax
    )
    ax.set_title("Temperature Anomaly vs Economic Impact by Disaster Type", fontsize=16, fontweight='bold')
    ax.set_ylabel("Economic Impact (USD)")
    ax.set_xlabel("Temperature Anomaly (°C)")
    
    # Add trend line - STATISTICAL CALCULATION
    slope, intercept, r_value, p_value, std_err = linregress(df['TemperatureAnomaly_C'], df['EconomicImpact_USD'])
    line_x = np.array([df['TemperatureAnomaly_C'].min(), df['TemperatureAnomaly_C'].max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r--', alpha=0.8, label=f'Trend (R² = {r_value**2:.3f})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # CO2 vs Temperature Correlation - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">CO₂ Levels vs Temperature Anomaly</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='CO2Level_ppm', y='TemperatureAnomaly_C', 
                   alpha=0.6, color='darkorange', s=60, ax=ax)
    
    # Add correlation info - STATISTICAL CALCULATION
    correlation = df['CO2Level_ppm'].corr(df['TemperatureAnomaly_C'])
    ax.set_title(f"CO₂ Levels vs Temperature Anomaly (Correlation: {correlation:.3f})", 
                fontsize=16, fontweight='bold')
    
    # Add trend line
    slope, intercept, r_value, p_value, std_err = linregress(df['CO2Level_ppm'], df['TemperatureAnomaly_C'])
    line_x = np.array([df['CO2Level_ppm'].min(), df['CO2Level_ppm'].max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
    
    st.pyplot(fig)
    
    # Country-wise Analysis - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Average Impact by Country</div>', unsafe_allow_html=True)
    
    country_impact = df.groupby('Country').agg({
        'EconomicImpact_USD': 'mean',
        'PopulationAffected': 'mean',
        'TemperatureAnomaly_C': 'mean'
    }).sort_values('EconomicImpact_USD', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(country_impact.index, country_impact['EconomicImpact_USD']/1e6, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(country_impact))))
    ax.set_title("Average Economic Impact by Country (Millions USD)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Average Economic Impact (Million USD)")
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, country_impact['EconomicImpact_USD']/1e6):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

# MULTIVARIATE ANALYSIS  
elif user_menu == 'Multivariate Analysis':
    st.markdown(f'<div class="main-header">{ANALYSIS_INFO["multivariate"]["title"]}</div>', unsafe_allow_html=True)
    st.markdown(ANALYSIS_INFO["multivariate"]["description"])
    
    # Correlation Matrix - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Climate Variables Correlation Matrix</div>', unsafe_allow_html=True)
    
    numeric_cols = ['TemperatureAnomaly_C', 'CO2Level_ppm', 'EconomicImpact_USD', 'PopulationAffected']
    correlation_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Climate Variables Correlation Matrix", fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    # 3D Scatter Plot - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">3D Climate Impact Visualization</div>', unsafe_allow_html=True)
    st.markdown(f"**{ANALYSIS_INFO['multivariate']['3d_description']}**")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(df['TemperatureAnomaly_C'], df['CO2Level_ppm'], df['EconomicImpact_USD'],
                        c=df['PopulationAffected'], cmap='viridis', alpha=0.7, s=60)
    
    ax.set_xlabel('Temperature Anomaly (°C)', fontsize=12, labelpad=10)
    ax.set_ylabel('CO₂ Level (ppm)', fontsize=12, labelpad=10)
    ax.set_zlabel('Economic Impact (USD)', fontsize=12, labelpad=10)
    ax.set_title("3D Climate Impact Analysis", fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, label='Population Affected', shrink=0.8)
    cbar.set_label('Population Affected', fontsize=12)
    st.pyplot(fig)
    
    # Statistical Summary - STATISTICAL CALCULATIONS
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variable Statistics:**")
        stats_df = df[numeric_cols].describe().round(2)
        st.dataframe(stats_df)
    
    with col2:
        st.markdown("**Country Impact Summary:**")
        country_summary = df.groupby('Country').agg({
            'EconomicImpact_USD': ['sum', 'mean'],
            'PopulationAffected': ['sum', 'mean'],
            'ExtremeWeatherEvent': 'count'
        }).round(2)
        country_summary.columns = ['Total Econ Impact', 'Avg Econ Impact', 'Total Pop Affected', 'Avg Pop Affected', 'Event Count']
        st.dataframe(country_summary.sort_values('Total Econ Impact', ascending=False))

# PREDICTION
elif user_menu == 'Prediction':
    st.markdown(f'<div class="main-header">{ANALYSIS_INFO["prediction"]["title"]}</div>', unsafe_allow_html=True)
    
    # Methodology Section - TEXTUAL FROM CONTENT.PY
    st.markdown(f'<div class="section-header">{PREDICTION_METHODOLOGY["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    <h4>Machine Learning Approach:</h4>
    <p>{PREDICTION_METHODOLOGY["overview"]}</p>
    
    <h4>Model Components:</h4>
    <ul>
    """, unsafe_allow_html=True)
    
    for component in PREDICTION_METHODOLOGY["components"]:
        st.markdown(f"<li><strong>{component}</strong></li>", unsafe_allow_html=True)
    
    st.markdown("""
    </ul>
    
    <h4>Input Features:</h4>
    <ul>
    """, unsafe_allow_html=True)
    
    for feature in PREDICTION_METHODOLOGY["features"]:
        st.markdown(f"<li><strong>{feature}</strong></li>", unsafe_allow_html=True)
    
    st.markdown(f"""
    </ul>
    
    <h4>Model Training:</h4>
    <p>{PREDICTION_METHODOLOGY["training_description"]} <strong>({len(df):,} records)</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Training - STATISTICAL CALCULATIONS
    @st.cache_data
    def prepare_models():
        def add_impact_level(df):
            econ_min = df['EconomicImpact_USD'].min()
            econ_max = df['EconomicImpact_USD'].max()
            pop_min = df['PopulationAffected'].min()
            pop_max = df['PopulationAffected'].max()
            
            df['EcoNorm'] = (df['EconomicImpact_USD'] - econ_min) / (econ_max - econ_min)
            df['PopNorm'] = (df['PopulationAffected'] - pop_min) / (pop_max - pop_min)
            df['CombinedImpactScore'] = df['EcoNorm'] + df['PopNorm']
            df['ImpactLevel'] = pd.qcut(df['CombinedImpactScore'], q=3, labels=['Low', 'Moderate', 'High'])
            return df
        
        # Prepare data
        model_df = add_impact_level(df.copy())
        en_country = LabelEncoder()
        model_df['CountryEnc'] = en_country.fit_transform(model_df['Country'])
        
        # Features
        features = ['CO2Level_ppm', 'TemperatureAnomaly_C', 'CountryEnc']
        X = model_df[features]
        
        # Models
        models = {}
        
        # Impact Level Classifier
        clf_impact = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        clf_impact.fit(X, model_df['ImpactLevel'])
        models['impact'] = clf_impact
        
        # Economic Impact Regressor
        reg_econ = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        reg_econ.fit(X, model_df['EconomicImpact_USD'])
        models['economic'] = reg_econ
        
        # Population Impact Regressor  
        reg_pop = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        reg_pop.fit(X, model_df['PopulationAffected'])
        models['population'] = reg_pop
        
        # Disaster Type Classifier
        clean_df = model_df[model_df['ExtremeWeatherEvent'].notna() & (model_df['ExtremeWeatherEvent'] != 'None')]
        if len(clean_df) > 0:
            clf_event = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            clf_event.fit(clean_df[features], clean_df['ExtremeWeatherEvent'])
            models['disaster_type'] = clf_event
        
        return models, en_country
    
    models, encoder = prepare_models()
    
    # Prediction Interface
    st.markdown('<div class="section-header">Prediction Interface</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"**{INTERFACE_LABELS['input_parameters']}**")
        
        input_country = st.selectbox(
            'Country:', 
            df['Country'].unique(), 
            help=INTERFACE_LABELS["country_help"]
        )
        
        # Calculate min/max values for sliders - STATISTICAL CALCULATIONS
        co2_min, co2_max = float(df['CO2Level_ppm'].min()), float(df['CO2Level_ppm'].max())
        co2_mean = float(df['CO2Level_ppm'].mean())
        temp_min, temp_max = float(df['TemperatureAnomaly_C'].min()), float(df['TemperatureAnomaly_C'].max())
        
        input_co2 = st.slider(
            'CO₂ Level (ppm):', 
            co2_min, co2_max,
            value=co2_mean,
            step=1.0,
            help=INTERFACE_LABELS["co2_help"]
        )
        
        input_temp = st.slider(
            'Temperature Anomaly (°C):', 
            temp_min, temp_max,
            value=0.0,
            step=0.1,
            help=INTERFACE_LABELS["temp_help"]
        )
        
        predict_button = st.button(INTERFACE_LABELS["predict_button"], type='primary')
    
    with col2:
        if predict_button:
            # Prepare prediction input - STATISTICAL CALCULATIONS
            input_country_enc = encoder.transform([input_country])[0]
            X_pred = np.array([[input_co2, input_temp, input_country_enc]])
            
            # Generate predictions - STATISTICAL CALCULATIONS
            pred_impact = models['impact'].predict(X_pred)[0]
            pred_economic = models['economic'].predict(X_pred)[0]
            pred_population = models['population'].predict(X_pred)[0]
            
            # Results display
            st.markdown(f"### {INTERFACE_LABELS['results_title']}")
            
            # Impact severity
            st.markdown(f"**{INTERFACE_LABELS['disaster_severity']}** **{pred_impact}**")
            
            # Economic and population impact
            col1_res, col2_res = st.columns(2)
            with col1_res:
                st.metric(INTERFACE_LABELS["economic_impact"], f"${pred_economic:,.0f}")
            with col2_res:
                st.metric(INTERFACE_LABELS["population_affected"], f"{int(pred_population):,}")
            
            # Disaster type prediction - STATISTICAL CALCULATIONS
            if 'disaster_type' in models:
                pred_proba = models['disaster_type'].predict_proba(X_pred)[0]
                event_labels = models['disaster_type'].classes_
                
                # Sort by probability
                prob_df = pd.DataFrame({
                    'Disaster Type': event_labels,
                    'Probability': pred_proba
                }).sort_values('Probability', ascending=False)
                
                st.markdown(f"**{INTERFACE_LABELS['likely_disasters']}**")
                for idx, row in prob_df.head(3).iterrows():
                    prob_percent = row['Probability'] * 100
                    st.markdown(f"• **{row['Disaster Type']}**: {prob_percent:.1f}% probability")
            
            # Risk Assessment - STATISTICAL CALCULATIONS
            st.markdown(f"### {INTERFACE_LABELS['risk_assessment']}")
            econ_75th_percentile = df['EconomicImpact_USD'].quantile(0.75)
            risk_level = "high" if pred_impact == "High" or pred_economic > econ_75th_percentile else "moderate" if pred_impact == "Moderate" else "low"
            
            if risk_level == "high":
                st.error(RISK_MESSAGES["high"])
            elif risk_level == "moderate":
                st.warning(RISK_MESSAGES["moderate"])
            else:
                st.success(RISK_MESSAGES["low"])

# DISASTER TYPES & SAFETY
elif user_menu == 'Disaster Types & Safety':
    st.markdown('<div class="main-header">Disaster Types & Safety Measures</div>', unsafe_allow_html=True)
    
    # Overview - TEXTUAL FROM CONTENT.PY
    st.markdown(f"""
    <div class="info-box">
    <h3>{DISASTER_GUIDE["overview_title"]}</h3>
    <p>{DISASTER_GUIDE["overview_text"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disaster Statistics - STATISTICAL CALCULATIONS
    st.markdown(f'<div class="section-header">{DISASTER_GUIDE["statistics_title"]}</div>', unsafe_allow_html=True)
    
    disaster_stats = df[df['ExtremeWeatherEvent'] != 'None']['ExtremeWeatherEvent'].value_counts()
    disaster_impact = df.groupby('ExtremeWeatherEvent').agg({
        'EconomicImpact_USD': 'mean',
        'PopulationAffected': 'mean'
    }).loc[disaster_stats.index]
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(disaster_stats.index, disaster_stats.values, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(disaster_stats))))
        ax.set_title("Frequency of Disaster Types", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(disaster_impact.index, disaster_impact['EconomicImpact_USD']/1e6, 
              color=plt.cm.Set2(np.linspace(0, 1, len(disaster_impact))))
        ax.set_title("Average Economic Impact by Disaster Type", fontsize=14, fontweight='bold')
        ax.set_ylabel("Economic Impact (Million USD)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Individual Disaster Types - TEXTUAL FROM CONTENT.PY + STATISTICAL CALCULATIONS
    for disaster_type, info in DISASTER_INFO.items():
        if disaster_type in df['ExtremeWeatherEvent'].values:
            # Get statistics for this disaster type - STATISTICAL CALCULATIONS
            disaster_data = df[df['ExtremeWeatherEvent'] == disaster_type]
            avg_economic = disaster_data['EconomicImpact_USD'].mean()
            avg_population = disaster_data['PopulationAffected'].mean()
            occurrences = len(disaster_data)
            
            st.markdown(f'<div class="disaster-card">', unsafe_allow_html=True)
            st.markdown(f'## {info["title"]}')
            st.markdown(f'**{info["description"]}**')
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Occurrences", f"{occurrences}")
            with col2:
                st.metric("Avg Economic Impact", f"${avg_economic/1e6:.1f}M")
            with col3:
                st.metric("Avg Population Affected", f"{int(avg_population):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Expandable sections for detailed info - TEXTUAL FROM CONTENT.PY
            with st.expander(f"Detailed Information for {disaster_type}"):
                tab1, tab2, tab3, tab4 = st.tabs(["Causes", "Before", "During", "After"])
                
                with tab1:
                    st.markdown("### Primary Causes")
                    for cause in info['causes']:
                        st.markdown(f"• {cause}")
                
                with tab2:
                    st.markdown("### Before the Disaster")
                    for tip in info['before']:
                        st.markdown(f'<div class="safety-tip">✓ {tip}</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("### During the Disaster")
                    for tip in info['during']:
                        st.markdown(f'<div class="warning-tip">! {tip}</div>', unsafe_allow_html=True)
                
                with tab4:
                    st.markdown("### After the Disaster")
                    for tip in info['after']:
                        st.markdown(f'<div class="emergency-tip">+ {tip}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Emergency Resources - TEXTUAL FROM CONTENT.PY
    st.markdown('<div class="section-header">Emergency Resources</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Emergency Contacts</h4>
        <ul>
        """, unsafe_allow_html=True)
        for contact in EMERGENCY_RESOURCES["contacts"]:
            st.markdown(f"<li><strong>{contact}</strong></li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Online Resources</h4>
        <ul>
        """, unsafe_allow_html=True)
        for resource in EMERGENCY_RESOURCES["online"]:
            st.markdown(f"<li><strong>{resource}</strong></li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
