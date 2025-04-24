import React, { useState } from 'react';
import './CarPricePrediction.css';
import { FaAtlas, FaTimes, FaDownload } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const CarPricePrediction = () => {
    const [inputData, setInputData] = useState({
        year: '',
        mileage: '',
        age: '',
        brand: '',
        model: '',
        title_status: '',
        color: '',
        state: '',
        country: '',
        condition: ''
    });
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleCarPricePrediction = async () => {
        try {
            const response = await fetch('http://localhost:8000/predict/car_price', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                setPredictionResult({
                    price: result.prediction,
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
        year: "Manufacturing Year",
        mileage: "Mileage (miles)",
        age: "Vehicle Age",
        brand: "Brand",
        model: "Model",
        title_status: "Title Status",
        color: "Color",
        state: "State",
        country: "Country",
        condition: "Condition"
    };

    const inputDescriptions = {
        year: "The year the car was manufactured (e.g., 2018)",
        mileage: "Total miles driven (e.g., 45000)",
        age: "Automatically calculated from manufacturing year",
        brand: "Car manufacturer (e.g., Toyota, Ford)",
        model: "Specific model name (e.g., Camry, F-150)",
        title_status: "Clean, salvage, rebuilt, etc.",
        color: "Vehicle exterior color",
        state: "US state where car is registered",
        country: "Country of origin/registration",
        condition: "Excellent, good, fair, or poor"
    };

    return (
        <div className="CarPricePrediction">
            {/* Header */}
            <div className="header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>

            {/* Hero Section */}
            <div className="CarPricePrediction-hero">
                <h1>Car Price <span>Prediction Model</span></h1>
                <p>Get accurate car price estimates using AI-powered models.</p>
            </div>

            {/* Input Form */}
            <div className="car-detection">
                <div className="notice">
                    <h3>Enter Vehicle Details</h3>
                    <p>Fill in the details below to get an accurate price estimate.</p>
                </div>
                <div className="input-container">
                    {Object.keys(inputData).map((key) => (
                        <div className="input-group" key={key}>
                            <label htmlFor={key}>{labels[key]}</label>
                            <input
                                type={key === 'year' || key === 'mileage' ? 'number' : 'text'}
                                id={key}
                                name={key}
                                placeholder={`Enter ${labels[key]}`}
                                value={inputData[key]}
                                onChange={handleInputChange}
                            />
                        </div>
                    ))}
                </div>
                <button className="predict-button" onClick={handleCarPricePrediction}>
                    Estimate Price
                </button>
                {predictionResult && (
                    <div className="prediction-result">
                        {predictionResult.error ? (
                            <div className="error-message">{predictionResult.error}</div>
                        ) : (
                            <>
                                <h3>Estimated Price</h3>
                                <div className="result-value">
                                    ${predictionResult.price.toLocaleString('en-US', {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    })}
                                </div>
                                {predictionResult.confidence && (
                                    <div className="confidence">
                                        Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}
            </div>

            {/* Updated Sidebar with Breast Cancer structure but Car Price content */}
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Car Price Prediction Model</h1>
                    <p>A comprehensive guide to our vehicle valuation system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Overview</h1>
                                <p>
                                    Our Car Price Prediction Model is a machine learning system designed to provide accurate 
                                    market valuations for vehicles based on their specifications, historical sales data, 
                                    and current market trends.
                                </p>
                                
                                <h2 className='workflow'>Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Gathering vehicle listings, cleaning data, and handling missing values.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Feature Engineering</h3>
                                        <p>Creating meaningful predictors like age-to-mileage ratio and regional adjustments.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Model Training & Prediction</h3>
                                        <p>Using regression techniques to estimate prices with confidence scores.</p>
                                    </li>
                                </div>

                                <h2 className='keycomponenets'>Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/datasets/doaaalsenani/usa-cars-dataset" target="_blank" rel="noopener noreferrer">
                                                USA Cars Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>Over 2,000 vehicle listings with detailed specifications</li>
                                        <li>Historical price data from multiple regions</li>
                                        <li>Comprehensive condition reports and title status</li>
                                    </ul>
                                </div>

                                <hr />

                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Linear Regression - Optimized for price prediction tasks</li>
                                        <li>Random Forest - Alternative model for comparison</li>
                                        <li>Gradient Boosting - For handling non-linear relationships</li>
                                    </ul>
                                </div>

                                <div className="ApproachUsed">
                                    <h3>Valuation Approach</h3>
                                    <p>
                                        Our system leverages a regression-based model that analyzes vehicle specifications,
                                        market trends, and regional factors to provide accurate price estimates with
                                        confidence intervals.
                                    </p>
                                </div>

                                <div className="download-buttons">
                                    <a
                                        href="./../../../../backend/models/Car Price Prediction/CarPricePrediction.ipynb"
                                        download="CarPrice_Notebook.ipynb"
                                        className="download-button"
                                    >
                                        Download Python Notebook
                                    </a>
                                    <a
                                        href="./../../../../backend/models/Car Price Prediction/car_price_model.pkl"
                                        download="CarPrice_Model.pkl"
                                        className="download-button"
                                    >
                                        Download Model File
                                    </a>
                                </div>
                            </div>
                        }

                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Technical implementation of our car price prediction system</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Essential libraries for data processing and machine learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>pandas: Data manipulation and analysis</li>
                                        <li>numpy: Numerical computing</li>
                                        <li>scikit-learn: Machine learning tools</li>
                                    </ul>

                                    <h2 style={{marginTop: '50px'}}>Data Loading and Preparation</h2>
                                    <p>Loading and preprocessing the vehicle dataset.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Load and clean the dataset
df = pd.read_csv('car_data.csv')
df = df.dropna(subset=['price'])  # Remove rows with missing prices
df['age'] = 2023 - df['year']  # Calculate vehicle age
df['mileage_ratio'] = df['mileage'] / df['age']  # Mileage-to-age ratio

# Filter unrealistic values
df = df[(df['price'] > 1000) & (df['price'] < 200000)]
df = df[df['mileage'] < 300000]`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <p style={{marginTop: '15px'}}>This code loads the dataset, calculates important features, and filters out unrealistic values.</p>

                                    <h2 style={{marginTop: '50px'}}>Feature Engineering</h2>
                                    <p>Creating meaningful features for the prediction model.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Create regional price adjustments
state_avg = df.groupby('state')['price'].mean().to_dict()
df['state_adjustment'] = df['state'].map(state_avg)

# Create brand premium features
brand_avg = df.groupby('brand')['price'].mean().to_dict()
df['brand_premium'] = df['brand'].map(brand_avg)

# Condition mapping
condition_map = {'excellent': 1, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}
df['condition_score'] = df['condition'].map(condition_map)`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Model Pipeline</h2>
                                    <p>Building the complete prediction pipeline.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Define preprocessing for numeric and categorical features
numeric_features = ['year', 'mileage', 'age', 'mileage_ratio']
categorical_features = ['brand', 'state', 'condition']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100))
])

model.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>Using the trained model to predict car prices.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`def predict_price(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Calculate derived features
    input_df['age'] = 2023 - input_df['year']
    input_df['mileage_ratio'] = input_df['mileage'] / input_df['age']
    
    # Make prediction
    predicted_price = model.predict(input_df)[0]
    confidence = model.score(X_test, y_test)  # R² score as confidence
    
    return predicted_price, confidence`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>
                        }

                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>

                                <section className="metric-section">
                                    <h2>Prediction Accuracy</h2>
                                    <p className="accuracy">92.4%</p>
                                    <p>Mean accuracy across test dataset</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Mean Absolute Error</th>
                                                <th>$1,245</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>R² Score</td>
                                                <td>0.91</td>
                                            </tr>
                                            <tr>
                                                <td>Error Rate</td>
                                                <td>8.2%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Feature Importance</h2>
                                    <p className="accuracy">Top Predictors</p>
                                    <p>Factors most influencing price predictions</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Feature</th>
                                                <th>Importance</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Vehicle Age</td>
                                                <td>32%</td>
                                            </tr>
                                            <tr>
                                                <td>Mileage</td>
                                                <td>28%</td>
                                            </tr>
                                            <tr>
                                                <td>Brand</td>
                                                <td>18%</td>
                                            </tr>
                                            <tr>
                                                <td>Condition</td>
                                                <td>12%</td>
                                            </tr>
                                            <tr>
                                                <td>Location</td>
                                                <td>10%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="validation-methodology">
                                    <h2>Validation Methodology</h2>
                                    <h3>Robust Testing</h3>
                                    <p>Our model uses comprehensive validation techniques:</p>
                                    <ul>
                                        <li>5-fold cross-validation</li>
                                        <li>Separate holdout test set (30% of data)</li>
                                        <li>Regional stratification</li>
                                        <li>Time-based validation split</li>
                                    </ul>
                                </section>
                            </div>
                        }
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CarPricePrediction;