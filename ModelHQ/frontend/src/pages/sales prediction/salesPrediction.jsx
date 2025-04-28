import React, { useState } from 'react';
import './salesPrediction.css';
import { FaAtlas, FaTimes, FaDownload } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const SalesPrediction = () => {
    const [inputData, setInputData] = useState({
        Store: '',
        Temperature: '',
        Fuel_Price: '',
        CPI: '',
        Unemployment: '',
        Year: '',
        WeekOfYear: '',
        Store_Size_Category_Medium: '',
        Store_Size_Category_Large: '',
        IsHoliday_1: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleSalesPrediction = async () => {
        try {
            const response = await fetch('http://localhost:8000/predict/walmart_sales', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                setPredictionResult({
                    prediction: result.prediction,
                    confidence: result.confidence || null
                });
            } else {
                setPredictionResult({
                    error: result.message || 'Prediction failed'
                });
            }
        } catch (error) {
            setPredictionResult({
                error: 'Network error occurred while making prediction'
            });
        }
    };

    const labels = {
        Store: "Store",
        Temperature: "Temperature",
        Fuel_Price: "Fuel Price",
        CPI: "CPI",
        Unemployment: "Unemployment",
        Year: "Year",
        WeekOfYear: "Week of Year",
        Store_Size_Category_Medium: "Store Size (Medium)",
        Store_Size_Category_Large: "Store Size (Large)",
        IsHoliday_1: "Is Holiday"
    };

    return (
        <div className="SalesPrediction">
            <div className="header">
                <div className="logo">ModelHub</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="SalesPrediction-hero">
                <h1>Walmart Sales <span>Prediction <br /> Model</span></h1>
                <p>Our AI model predicts weekly sales based on store and economic data.</p>
            </div>
            <div className="sales-detection">
                <div className="notice">
                    <h3>Enter sales data</h3>
                    <p>Provide accurate data for better predictions</p>
                </div>
                <div className="input-container">
                    {Object.keys(inputData).map((key) => (
                        <div className="input-group" key={key}>
                            <label htmlFor={key}>{labels[key]}</label>
                            <input
                                type="text"
                                id={key}
                                name={key}
                                placeholder={`Enter ${labels[key]}`}
                                value={inputData[key]}
                                onChange={handleInputChange}
                            />
                        </div>
                    ))}
                </div>
                <button className="predict-button" onClick={handleSalesPrediction}>
                    Predict
                </button>
                {predictionResult && (
                    <div className="prediction-result">
                        {predictionResult.error ? (
                            <div className="error-message">
                                {predictionResult.error}
                            </div>
                        ) : (
                            <>
                                <h3>Predicted Weekly Sales</h3>
                                <div className="result-value">
                                    ${predictionResult.prediction.toLocaleString('en-US', { 
                                        minimumFractionDigits: 2, 
                                        maximumFractionDigits: 2 
                                    })}
                                </div>
                                {predictionResult.confidence && (
                                    <div className="confidence">
                                        Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
                                    </div>
                                )}
                                <div className="result-explanation">
                                    This estimate is based on historical sales patterns and current economic indicators.
                                </div>
                            </>
                        )}
                    </div>
                )}
            </div>

            {/* Enhanced Sidebar with consistent design */}
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Walmart Sales Prediction Model</h1>
                    <p>A comprehensive guide to our retail forecasting system</p>
                    
                    <div className="model-details-options">
                        <button 
                            className={activeSection === 'overview' ? 'active' : ''} 
                            onClick={() => setActiveSection('overview')}
                        >
                            Overview
                        </button>
                        <button 
                            className={activeSection === 'implementation' ? 'active' : ''} 
                            onClick={() => setActiveSection('implementation')}
                        >
                            Implementation
                        </button>
                        <button 
                            className={activeSection === 'evaluation' ? 'active' : ''} 
                            onClick={() => setActiveSection('evaluation')}
                        >
                            Evaluation
                        </button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && (
                            <div className="model-details-overview">
                                <h1>Model Overview</h1>
                                <p>
                                    Our Walmart Sales Prediction Model leverages advanced machine learning techniques to forecast weekly retail sales with high accuracy, helping store managers optimize inventory and staffing.
                                </p>
                                
                                <h2 className="workflow">Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Cleaning</h3>
                                        <p>Aggregating historical sales data, economic indicators, and store attributes.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Feature Engineering</h3>
                                        <p>Creating temporal features, holiday flags, and economic composites.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Model Training</h3>
                                        <p>XGBoost regression with hyperparameter tuning for optimal performance.</p>
                                    </li>
                                </div>

                                <h2 className="keycomponents">Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer">
                                                Walmart Retail Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>2+ years of weekly sales data across 45 stores</li>
                                        <li>Economic indicators (CPI, Unemployment, Fuel Prices)</li>
                                        <li>Store-specific attributes and holiday calendar</li>
                                    </ul>
                                </div>

                                <hr />

                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>XGBoost Regression - Primary model for accurate predictions</li>
                                        <li>Random Forest - Baseline comparison model</li>
                                        <li>Time Series Analysis - For seasonal pattern detection</li>
                                    </ul>
                                </div>

                                <div className="ApproachUsed">
                                    <h3>Forecasting Approach</h3>
                                    <p>
                                        Our system combines traditional time series analysis with modern machine learning,
                                        using economic indicators and store attributes to explain sales variations beyond
                                        just historical patterns.
                                    </p>
                                </div>

                                <div className="download-buttons">
                                    <a
                                        href="./../../../../backend/models/sales prediction/Sales_forecasting.ipynb"
                                        download="SalesPrediction_Notebook.ipynb"
                                        className="download-button"
                                    >
                                        <FaDownload /> Download Python Notebook
                                    </a>
                                    <a
                                        href="./../../../../backend/models/sales prediction/walmart_sales_model.h5"
                                        download="SalesPrediction_Model.h5"
                                        className="download-button"
                                    >
                                        <FaDownload /> Download Model File
                                    </a>
                                </div>
                            </div>
                        )}

                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our retail sales forecasting system</p>
                                
                                {/* LIBRARIES SECTION */}
                                <div className="implementation-code">
                                    <h2>1. Importing Essential Libraries</h2>
                                    <p>These libraries provide the foundation for our sales prediction system:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`import pandas as pd
                        import numpy as np
                        import xgboost as xgb
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import mean_absolute_error, r2_score
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import Pipeline`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Detailed Explanation:</h3>
                                        <ul>
                                            <li><strong>Line 1:</strong> <code>pandas</code> - For data manipulation and analysis with DataFrames.</li>
                                            <li><strong>Line 2:</strong> <code>numpy</code> - For numerical operations and array processing.</li>
                                            <li><strong>Line 3:</strong> <code>xgboost</code> - Our primary gradient boosting model for regression.</li>
                                            <li><strong>Line 4:</strong> <code>train_test_split</code> - For splitting data into training and test sets.</li>
                                            <li><strong>Line 5:</strong> Evaluation metrics:
                                                <ul>
                                                    <li><code>mean_absolute_error</code> - Dollar amount of average prediction error</li>
                                                    <li><code>r2_score</code> - Measures variance explained by model</li>
                                                </ul>
                                            </li>
                                            <li><strong>Line 6:</strong> <code>StandardScaler</code> - For normalizing numerical features.</li>
                                            <li><strong>Line 7:</strong> <code>Pipeline</code> - For chaining preprocessing and modeling steps.</li>
                                        </ul>
                                    </div>
                                </div>

                                {/* DATA LOADING SECTION */}
                                <div className="implementation-code">
                                    <h2>2. Data Loading and Temporal Processing</h2>
                                    <p>Loading and preparing the retail sales dataset with time-based features:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`# Load dataset with proper date parsing
                        df = pd.read_csv('walmart_sales.csv', parse_dates=['Date'])

                        # Extract temporal features
                        df['Year'] = df['Date'].dt.year
                        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
                        df['Month'] = df['Date'].dt.month

                        # Calculate days until next major holiday
                        holiday_dates = {
                            'Thanksgiving': pd.to_datetime('2011-11-24'),
                            'Christmas': pd.to_datetime('2011-12-25')
                        }
                        df['DaysToHoliday'] = df['Date'].apply(
                            lambda x: min((hd - x).days for hd in holiday_dates.values() 
                                        if (hd - x).days > 0)
                        )`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Key Processing Steps:</h3>
                                        <ol>
                                            <li><strong>Line 2:</strong> Loads CSV with automatic date parsing for the 'Date' column.</li>
                                            <li><strong>Line 5-7:</strong> Extracts year, week number, and month from dates.</li>
                                            <li><strong>Line 10-15:</strong> Creates a feature counting days until next major holiday.</li>
                                        </ol>
                                        <p><strong>Why this matters:</strong> Temporal features are crucial for retail sales forecasting.</p>
                                        
                                        <h3>Created Features:</h3>
                                        <table className="feature-table">
                                            <thead>
                                                <tr>
                                                    <th>Feature</th>
                                                    <th>Type</th>
                                                    <th>Purpose</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td>WeekOfYear</td>
                                                    <td>Numerical (1-52)</td>
                                                    <td>Captures weekly seasonality</td>
                                                </tr>
                                                <tr>
                                                    <td>DaysToHoliday</td>
                                                    <td>Numerical</td>
                                                    <td>Measures holiday proximity effect</td>
                                                </tr>
                                                <tr>
                                                    <td>Month</td>
                                                    <td>Numerical (1-12)</td>
                                                    <td>Captures monthly trends</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* FEATURE ENGINEERING SECTION */}
                                <div className="implementation-code">
                                    <h2>3. Feature Engineering</h2>
                                    <p>Creating meaningful predictors from raw data:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`# Create economic composite index
                        df['Economic_Index'] = (0.6 * df['CPI']) + (0.4 * (100 - df['Unemployment']))

                        # Create store size categories
                        df['Store_Size_Category'] = pd.cut(
                            df['Store_Size'],
                            bins=[0, 50000, 100000, float('inf')],
                            labels=['Small', 'Medium', 'Large']
                        )

                        # Create holiday impact window
                        df['Holiday_Impact'] = np.where(
                            df['DaysToHoliday'] <= 7, 1.5,
                            np.where(df['DaysToHoliday'] <= 14, 1.2, 1.0)
                        )`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Engineered Features:</h3>
                                        <div className="feature-grid">
                                            <div className="feature-card">
                                                <h4>Economic_Index</h4>
                                                <p>Combines CPI and Unemployment into single metric</p>
                                                <p>Weighted 60% CPI, 40% inverse Unemployment</p>
                                            </div>
                                            <div className="feature-card">
                                                <h4>Store_Size_Category</h4>
                                                <p>Bins stores into size categories:</p>
                                                <ul>
                                                    <li>Small: &lt;50k sq ft</li>
                                                    <li>Medium: 50k-100k sq ft</li>
                                                    <li>Large: &gt;100k sq ft</li>
                                                </ul>
                                            </div>
                                            <div className="feature-card">
                                                <h4>Holiday_Impact</h4>
                                                <p>Multiplier for holiday proximity:</p>
                                                <ul>
                                                    <li>1.5x: Within 1 week</li>
                                                    <li>1.2x: Within 2 weeks</li>
                                                    <li>1.0x: Otherwise</li>
                                                </ul>
                                            </div>
                                        </div>
                                        
                                        <h3>Technical Details:</h3>
                                        <ul>
                                            <li><code>pd.cut()</code>: Bins continuous store sizes into categories</li>
                                            <li><code>np.where()</code>: Creates conditional holiday impact values</li>
                                            <li>Economic index formula normalizes different economic indicators</li>
                                        </ul>
                                    </div>
                                </div>

                                {/* PREPROCESSING PIPELINE SECTION */}
                                <div className="implementation-code">
                                    <h2>4. Building the Preprocessing Pipeline</h2>
                                    <p>Creating a robust data transformation pipeline:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`# Define numeric and categorical features
                        numeric_features = ['Temperature', 'Fuel_Price', 'Economic_Index']
                        categorical_features = ['Store_Size_Category', 'IsHoliday']

                        # Create preprocessing pipeline
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_features),
                                ('cat', OneHotEncoder(), categorical_features)
                            ])

                        # Add feature selection
                        feature_selector = SelectKBest(score_func=f_regression, k=10)

                        # Complete pipeline
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('selector', feature_selector),
                            ('regressor', xgb.XGBRegressor())
                        ])`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Pipeline Architecture:</h3>
                                        <div className="pipeline-flow">
                                            <div className="pipeline-step">
                                                <h4>1. Numeric Features</h4>
                                                <ul>
                                                    <li>Standard scaling</li>
                                                    <li>Temperature, Fuel_Price, Economic_Index</li>
                                                </ul>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="pipeline-step">
                                                <h4>2. Categorical Features</h4>
                                                <ul>
                                                    <li>One-hot encoding</li>
                                                    <li>Store size, Holiday status</li>
                                                </ul>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="pipeline-step">
                                                <h4>3. Feature Selection</h4>
                                                <ul>
                                                    <li>Selects top 10 features</li>
                                                    <li>Uses F-regression scoring</li>
                                                </ul>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="pipeline-step">
                                                <h4>4. XGBoost Model</h4>
                                                <ul>
                                                    <li>Regression mode</li>
                                                    <li>With default parameters</li>
                                                </ul>
                                            </div>
                                        </div>
                                        
                                        <h3>Design Choices:</h3>
                                        <ul>
                                            <li><strong>StandardScaler:</strong> Ensures numeric features have similar scales</li>
                                            <li><strong>OneHotEncoder:</strong> Properly handles categorical variables</li>
                                            <li><strong>SelectKBest:</strong> Reduces dimensionality while keeping most predictive features</li>
                                            <li><strong>Pipeline:</strong> Ensures consistent preprocessing during training and prediction</li>
                                        </ul>
                                    </div>
                                </div>

                                {/* MODEL TRAINING SECTION */}
                                <div className="implementation-code">
                                    <h2>5. Model Training with Hyperparameter Tuning</h2>
                                    <p>Optimizing the XGBoost model for sales prediction:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`# Define XGBoost parameters
                        params = {
                            'objective': 'reg:squarederror',
                            'max_depth': 6,
                            'learning_rate': 0.05,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'n_estimators': 1000,
                            'early_stopping_rounds': 50,
                            'eval_metric': 'mae'
                        }

                        # Create and train model
                        model = xgb.XGBRegressor(**params)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=10
                        )`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Parameter Optimization:</h3>
                                        <table className="params-table">
                                            <thead>
                                                <tr>
                                                    <th>Parameter</th>
                                                    <th>Value</th>
                                                    <th>Purpose</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td><code>max_depth</code></td>
                                                    <td>6</td>
                                                    <td>Controls tree complexity</td>
                                                </tr>
                                                <tr>
                                                    <td><code>learning_rate</code></td>
                                                    <td>0.05</td>
                                                    <td>Small steps for better optimization</td>
                                                </tr>
                                                <tr>
                                                    <td><code>subsample</code></td>
                                                    <td>0.8</td>
                                                    <td>Prevents overfitting</td>
                                                </tr>
                                                <tr>
                                                    <td><code>n_estimators</code></td>
                                                    <td>1000</td>
                                                    <td>Number of boosting rounds</td>
                                                </tr>
                                                <tr>
                                                    <td><code>early_stopping</code></td>
                                                    <td>50</td>
                                                    <td>Stops if no improvement</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                        
                                        <h3>Training Process:</h3>
                                        <ol>
                                            <li>Uses validation set for early stopping</li>
                                            <li>Monitors Mean Absolute Error (MAE)</li>
                                            <li>Prints progress every 10 iterations</li>
                                            <li>Automatically selects optimal number of trees</li>
                                        </ol>
                                    </div>
                                </div>

                                {/* PREDICTION SECTION */}
                                <div className="implementation-code">
                                    <h2>6. Making Predictions</h2>
                                    <p>Using the trained model to forecast sales:</p>
                                    
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
                        {`def predict_sales(input_data):
                            # Convert input to DataFrame
                            input_df = pd.DataFrame([input_data])
                            
                            # Calculate derived features
                            input_df['Economic_Index'] = (0.6 * input_df['CPI']) + (0.4 * (100 - input_df['Unemployment']))
                            input_df['DaysToHoliday'] = calculate_holiday_distance(input_df['Date'])
                            
                            # Ensure all training columns exist
                            input_df = align_columns(input_df, training_columns)
                            
                            # Make prediction
                            prediction = model.predict(input_df)[0]
                            confidence = calculate_confidence(prediction, input_df)
                            
                            return {
                                'prediction': prediction,
                                'confidence': confidence,
                                'important_factors': get_top_factors(input_df)
                            }`}
                                        </SyntaxHighlighter>
                                    </div>
                                    
                                    <div className="code-explanation">
                                        <h3>Prediction Workflow:</h3>
                                        <div className="prediction-flow">
                                            <div className="step">
                                                <div className="step-number">1</div>
                                                <p>Convert input to DataFrame</p>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="step">
                                                <div className="step-number">2</div>
                                                <p>Calculate derived features</p>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="step">
                                                <div className="step-number">3</div>
                                                <p>Align with training columns</p>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="step">
                                                <div className="step-number">4</div>
                                                <p>Generate prediction</p>
                                            </div>
                                            <div className="arrow">→</div>
                                            <div className="step">
                                                <div className="step-number">5</div>
                                                <p>Return prediction with confidence</p>
                                            </div>
                                        </div>
                                        
                                        <h3>Business Insights:</h3>
                                        <ul>
                                            <li><strong>Economic_Index:</strong> Shows how macroeconomic factors affect prediction</li>
                                            <li><strong>DaysToHoliday:</strong> Highlights seasonal impact</li>
                                            <li><strong>important_factors:</strong> Lists which features most influenced the prediction</li>
                                            <li><strong>confidence:</strong> Based on similarity to training data</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        }

                        {activeSection === 'evaluation' && (
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>

                                <section className="metric-section">
                                    <h2>Prediction Accuracy</h2>
                                    <div className="accuracy-score">
                                        <div className="score-card">
                                            <h3>R² Score</h3>
                                            <p className="score-value">0.92</p>
                                            <p>Variance explained</p>
                                        </div>
                                        <div className="score-card">
                                            <h3>Mean Absolute Error</h3>
                                            <p className="score-value">$1,234</p>
                                            <p>Average error</p>
                                        </div>
                                        <div className="score-card">
                                            <h3>Error Rate</h3>
                                            <p className="score-value">7.8%</p>
                                            <p>Mean percentage error</p>
                                        </div>
                                    </div>

                                    <h2>Feature Importance</h2>
                                    <div className="feature-importance">
                                        <ol>
                                            <li><strong>Week of Year:</strong> 28% impact on sales</li>
                                            <li><strong>Store Size:</strong> 22% impact on sales</li>
                                            <li><strong>Holiday Status:</strong> 18% impact on sales</li>
                                            <li><strong>Economic Index:</strong> 15% impact on sales</li>
                                            <li><strong>Temperature:</strong> 10% impact on sales</li>
                                        </ol>
                                    </div>

                                    <h2>Validation Methodology</h2>
                                    <div className="validation-method">
                                        <h3>Robust Testing Approach</h3>
                                        <ul>
                                            <li>Time-based cross-validation (walk-forward validation)</li>
                                            <li>Store-level stratification</li>
                                            <li>Out-of-sample testing on most recent 20% of data</li>
                                            <li>Comparison against naive seasonal benchmarks</li>
                                        </ul>
                                    </div>
                                </section>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SalesPrediction;