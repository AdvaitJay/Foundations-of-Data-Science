"""
Content Module for Climate Change Impact Dashboard
Contains only textual information - descriptions, methodology, safety measures.
Statistical calculations remain in app.py.
"""

# Dashboard Information
DASHBOARD_INFO = {
    "title": "Climate Change Impact Dashboard",
    "description": """
    Climate Change Impact Dashboard is a comprehensive analytical tool designed to visualize 
    and understand the profound impacts of climate change on our planet. This dashboard analyzes 
    real-world data spanning from 2000 to 2023, covering major climate disasters across multiple countries.
    """,
    "objectives": [
        "Data Visualization: Interactive charts and graphs showing climate trends and disaster patterns",
        "Impact Assessment: Economic and human cost analysis of climate disasters", 
        "Predictive Modeling: Machine learning-powered disaster prediction and impact estimation",
        "Safety Awareness: Comprehensive disaster preparedness and safety information",
        "Evidence-Based Insights: Supporting climate action with concrete data"
    ],
    "dataset_metrics": [
        "Temperature Anomaly: Deviations from normal temperature (°C)",
        "CO₂ Levels: Atmospheric carbon dioxide concentration (ppm)",
        "Economic Impact: Financial damage in USD",
        "Population Affected: Number of people impacted",
        "Disaster Types: Classification of extreme weather events"
    ],
    "navigation_sections": [
        "Univariate Analysis: Individual variable distributions and patterns",
        "Bivariate Analysis: Relationships between two variables",
        "Multivariate Analysis: Complex interactions between multiple variables",
        "Prediction: AI-powered disaster prediction and impact estimation",
        "Disaster Types & Safety: Comprehensive disaster information and safety measures"
    ]
}

# Analysis Descriptions
ANALYSIS_INFO = {
    "univariate": {
        "title": "Univariate Analysis",
        "description": "Explore individual variable distributions and patterns to understand climate data characteristics.",
        "temperature_insight": "Temperature anomalies show how much temperatures deviate from historical norms."
    },
    "bivariate": {
        "title": "Bivariate Analysis", 
        "description": "Explore relationships and correlations between different climate variables."
    },
    "multivariate": {
        "title": "Multivariate Analysis",
        "description": "Analyze complex relationships between multiple climate variables simultaneously.",
        "3d_description": "Interactive 3D view of Temperature, CO₂, Economic Impact, and Population affected"
    },
    "prediction": {
        "title": "Climate Disaster Prediction",
        "description": "Advanced AI-powered prediction system for climate disaster impacts."
    }
}

# Prediction Methodology
PREDICTION_METHODOLOGY = {
    "title": "Advanced AI Methodology",
    "overview": """
    Our prediction system employs Random Forest algorithms, which are ensemble learning methods 
    that combine multiple decision trees to create robust, accurate predictions.
    """,
    "components": [
        "Impact Severity Classifier: Predicts disaster impact level (Low, Moderate, High)",
        "Economic Impact Regressor: Estimates financial damage in USD",
        "Population Impact Regressor: Predicts number of people affected",
        "Disaster Type Classifier: Identifies most likely disaster type with probability scores"
    ],
    "features": [
        "Geographic Location: Country-specific risk patterns encoded using label encoding",
        "Atmospheric CO₂ Levels: Current greenhouse gas concentrations (ppm)",
        "Temperature Anomaly: Deviation from historical temperature norms (°C)"
    ],
    "training_description": "Models are trained on historical climate disaster records spanning 2000-2023, ensuring predictions are based on real-world patterns and outcomes."
}

# Disaster Information
DISASTER_INFO = {
    'Flood': {
        'title': 'Flood',
        'description': 'Floods occur when water overflows onto normally dry land. They can be caused by heavy rainfall, river overflow, coastal storms, or dam failures.',
        'causes': [
            'Excessive rainfall or snowmelt',
            'River overflow due to blockages',
            'Coastal storm surges',
            'Dam or levee failures',
            'Poor urban drainage systems'
        ],
        'before': [
            'Create a family emergency plan with evacuation routes',
            'Identify higher ground locations near your home',
            'Keep emergency supplies (water, food, flashlights, battery radio)',
            'Learn to turn off utilities (gas, electricity, water)',
            'Stay informed through weather alerts and warnings',
            'Consider flood insurance (takes 30 days to take effect)'
        ],
        'during': [
            'Move to higher ground immediately',
            'Avoid walking or driving through flood waters',
            'Never drive around road barriers',
            'Stay away from downed power lines',
            'Listen to emergency broadcasts',
            'Call for help if trapped, but do not enter flood water'
        ],
        'after': [
            'Wait for authorities to declare area safe',
            'Avoid flood water - may contain sewage or chemicals',
            'Check for structural damage before entering buildings',
            'Document damage with photos for insurance',
            'Throw away contaminated food and water',
            'Boil water until authorities declare it safe'
        ]
    },
    'Drought': {
        'title': 'Drought',
        'description': 'Drought is a prolonged period of abnormally low rainfall, leading to water shortages that affect agriculture, water supply, and ecosystems.',
        'causes': [
            'Extended periods of below-normal precipitation',
            'High temperatures increasing evaporation',
            'Changes in atmospheric circulation patterns',
            'Climate change altering weather patterns',
            'Overuse of water resources'
        ],
        'before': [
            'Implement water conservation measures',
            'Install water-efficient appliances and fixtures',
            'Create drought-resistant landscaping',
            'Store emergency water supplies',
            'Develop alternative water sources',
            'Monitor local water restrictions and alerts'
        ],
        'during': [
            'Follow all water use restrictions strictly',
            'Prioritize water for drinking and essential needs',
            'Use greywater for non-potable purposes',
            'Protect remaining water sources from contamination',
            'Monitor vulnerable populations (elderly, children)',
            'Consider temporary relocation if water becomes scarce'
        ],
        'after': [
            'Continue water conservation practices',
            'Assess and repair any damaged water infrastructure',
            'Replant with drought-resistant vegetation',
            'Evaluate water management strategies',
            'Support community recovery efforts',
            'Plan for future drought resilience'
        ]
    },
    'Wildfire': {
        'title': 'Wildfire',
        'description': 'Wildfires are uncontrolled fires that spread rapidly through vegetation. They can be natural or human-caused and are intensified by dry conditions and strong winds.',
        'causes': [
            'Lightning strikes in dry conditions',
            'Human activities (campfires, cigarettes, arson)',
            'Power line failures',
            'Vehicle accidents or malfunctions',
            'Drought conditions creating dry fuel',
            'Strong winds spreading flames rapidly'
        ],
        'before': [
            'Create defensible space around your home',
            'Use fire-resistant building materials',
            'Maintain clear evacuation routes',
            'Prepare emergency supply kits for quick evacuation',
            'Sign up for local emergency alerts',
            'Have multiple evacuation routes planned'
        ],
        'during': [
            'Evacuate immediately when ordered',
            'If trapped, call 911 and give location',
            'Stay indoors with windows closed if unable to evacuate',
            'Use N95 masks to protect from smoke',
            'Monitor emergency broadcasts continuously',
            'Never ignore evacuation orders'
        ],
        'after': [
            'Wait for official all-clear before returning',
            'Check for hot spots and smoldering debris',
            'Inspect utilities before use',
            'Document damage for insurance claims',
            'Be aware of unstable trees and structures',
            'Watch for flash flooding in burned areas'
        ]
    },
    'Hurricane': {
        'title': 'Hurricane',
        'description': 'Hurricanes are intense tropical cyclones with sustained winds of 74+ mph. They bring destructive winds, storm surge, and heavy rainfall.',
        'causes': [
            'Warm ocean temperatures (≥26.5°C)',
            'Low atmospheric pressure systems',
            'Minimal wind shear in upper atmosphere',
            'Sufficient distance from the equator',
            'Pre-existing weather disturbance',
            'Climate change intensifying storms'
        ],
        'before': [
            'Develop family emergency and evacuation plans',
            'Stock emergency supplies for at least 7 days',
            'Secure outdoor furniture and objects',
            'Install storm shutters or board up windows',
            'Know your evacuation zone and routes',
            'Keep important documents in waterproof container'
        ],
        'during': [
            'Stay indoors away from windows',
            'If evacuating, leave early before conditions worsen',
            'Never go outside during the eye of the storm',
            'Listen to weather updates continuously',
            'Avoid using candles - use flashlights instead',
            'Stay on upper floors if flooding occurs'
        ],
        'after': [
            'Wait for official all-clear signal',
            'Avoid downed power lines and flood water',
            'Check for injuries and provide first aid',
            'Inspect home for damage before full occupancy',
            'Take photos of damage for insurance',
            'Be patient - restoration may take weeks'
        ]
    },
    'Tornado': {
        'title': 'Tornado',
        'description': 'Tornadoes are violently rotating columns of air extending from thunderstorms to the ground, with winds that can exceed 300 mph.',
        'causes': [
            'Severe thunderstorms with rotating updrafts',
            'Temperature and humidity contrasts',
            'Wind shear at different altitudes',
            'Supercell thunderstorm development',
            'Atmospheric instability',
            'Topographical influences'
        ],
        'before': [
            'Identify safe rooms in your home (interior, lowest floor)',
            'Practice tornado drills with family',
            'Keep emergency supplies in safe areas',
            'Install a weather alert radio',
            'Know the difference between watches and warnings',
            'Have sturdy shoes available for aftermath'
        ],
        'during': [
            'Go to safe room immediately when warning issued',
            'Get as low as possible, cover head and neck',
            'Stay away from windows and large roof spans',
            'In vehicle: abandon and seek low ground',
            'In mobile home: leave immediately for sturdy shelter',
            'Never try to outrun a tornado in urban areas'
        ],
        'after': [
            'Check for injuries and provide first aid',
            'Be careful of sharp debris and broken glass',
            'Stay out of damaged buildings',
            'Avoid downed power lines',
            'Listen for emergency information',
            'Help neighbors who may need assistance'
        ]
    },
    'Heatwave': {
        'title': 'Heatwave',
        'description': 'Heatwaves are prolonged periods of excessively hot weather, often accompanied by high humidity, that can be dangerous to human health.',
        'causes': [
            'High-pressure systems trapping hot air',
            'Urban heat island effects',
            'Climate change increasing frequency',
            'Jet stream patterns',
            'Lack of cooling rainfall',
            'Drought conditions intensifying heat'
        ],
        'before': [
            'Identify air-conditioned places (cooling centers)',
            'Install weather stripping and insulation',
            'Service air conditioning systems',
            'Plan for vulnerable family members',
            'Stock up on water and electrolyte solutions',
            'Learn to recognize heat-related illness symptoms'
        ],
        'during': [
            'Stay indoors during hottest hours (10am-6pm)',
            'Drink water frequently, avoid alcohol',
            'Wear lightweight, light-colored clothing',
            'Take cool showers or baths',
            'Check on elderly and vulnerable neighbors',
            'Never leave people or pets in vehicles'
        ],
        'after': [
            'Continue monitoring vulnerable individuals',
            'Assess any heat-related damage to property',
            'Evaluate cooling system performance',
            'Restock emergency supplies',
            'Review and improve heat preparedness plans',
            'Support community recovery if infrastructure damaged'
        ]
    }
}

# Emergency Resources
EMERGENCY_RESOURCES = {
    "contacts": [
        "Emergency Services: 911 (US/Canada)",
        "Red Cross: 1-800-RED-CROSS",
        "FEMA: 1-800-621-3362",
        "Poison Control: 1-800-222-1222"
    ],
    "online": [
        "Ready.gov: Official preparedness guide",
        "Weather.gov: National Weather Service", 
        "RedCross.org: Disaster preparedness",
        "FEMA.gov: Federal disaster information"
    ]
}

# Risk Assessment Messages
RISK_MESSAGES = {
    "high": "HIGH RISK: Immediate disaster preparedness recommended",
    "moderate": "MODERATE RISK: Enhanced monitoring and preparation advised",
    "low": "LOW RISK: Standard monitoring sufficient"
}

# Interface Labels
INTERFACE_LABELS = {
    "input_parameters": "Select Input Parameters:",
    "country_help": "Select the country for prediction analysis",
    "co2_help": "Atmospheric CO₂ concentration in parts per million",
    "temp_help": "Temperature deviation from historical normal",
    "predict_button": "Generate Predictions",
    "results_title": "Prediction Results",
    "disaster_severity": "Disaster Severity:",
    "economic_impact": "Economic Impact",
    "population_affected": "Population Affected",
    "likely_disasters": "Most Likely Disaster Types:",
    "risk_assessment": "Risk Assessment"
}

# Warning Messages
WARNING_MESSAGES = {
    "temp_rising": "WARNING: Global temperatures are rising above normal levels",
    "co2_high": "WARNING: CO₂ levels are critically high (>400 ppm)"
}

# Disaster Guide Overview
DISASTER_GUIDE = {
    "overview_title": "Comprehensive Disaster Preparedness Guide",
    "overview_text": "Understanding different types of climate disasters and their safety measures is crucial for protecting lives and property. This guide provides detailed information about each disaster type found in our climate data, along with evidence-based safety recommendations.",
    "statistics_title": "Disaster Statistics from Our Data"
}