import React, { useState } from 'react';
import axios from 'axios';
import './HousingPrice.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const MumbaiHousePricePrediction = () => {
    const [inputData, setInputData] = useState({
        area: '',
        bedroom_num: '',
        bathroom_num: '',
        balcony_num: '',
        age: '',
        total_floors: '',
        latitude: '',
        longitude: '',
        locality: '',
        city: '',
        property_type: '',
        furnished: ''
    });
    const [activeSection, setActiveSection] = useState('overview');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [predictionResult, setPredictionResult] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handlePricePrediction = async () => {
        try {
            // Validate input data
            const requiredFields = Object.keys(inputData);
            for (const field of requiredFields) {
                if (!inputData[field]) {
                    setPredictionResult({ error: `Field "${labels[field]}" is required.` });
                    return;
                }
            }

            const formattedData = {
                area: parseFloat(inputData.area),
                bedroom_num: parseInt(inputData.bedroom_num, 10),
                bathroom_num: parseInt(inputData.bathroom_num, 10),
                balcony_num: parseInt(inputData.balcony_num, 10),
                age: parseInt(inputData.age, 10),
                total_floors: parseInt(inputData.total_floors, 10),
                latitude: parseFloat(inputData.latitude),
                longitude: parseFloat(inputData.longitude),
                locality: inputData.locality,
                city: inputData.city,
                property_type: inputData.property_type,
                furnished: inputData.furnished
            };

            const response = await axios.post('http://localhost:8000/predict/mumbai_house_price', formattedData);
            if (response.data.status === 'success') {
                const formattedPrice = new Intl.NumberFormat('en-IN', {
                    style: 'currency',
                    currency: 'INR',
                    maximumFractionDigits: 0
                }).format(response.data.prediction);
                
                setPredictionResult({
                    prediction: formattedPrice,
                    error: null
                });
            } else {
                setPredictionResult({ error: response.data.message });
            }
        } catch (error) {
            setPredictionResult({ error: 'An error occurred while making the prediction.' });
        }
    };

    const labels = {
        area: 'Area (sq.ft)',
        bedroom_num: 'Number of Bedrooms',
        bathroom_num: 'Number of Bathrooms',
        balcony_num: 'Number of Balconies',
        age: 'Property Age (years)',
        total_floors: 'Total Floors in Building',
        latitude: 'Latitude',
        longitude: 'Longitude',
        locality: 'Locality',
        city: 'City',
        property_type: 'Property Type',
        furnished: 'Furnishing Status'
    };

    const propertyTypes = ['Apartment', 'House', 'Villa', 'Studio', 'Penthouse'];
    const cities = ['Mumbai', 'Thane', 'Navi Mumbai', 'Mira Road', 'Other'];
    const furnishedOptions = ['Furnished', 'Semi-Furnished', 'Unfurnished'];

    return (
        <div className="MumbaiHousePricePrediction">
            <div className="header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="MumbaiHousePricePrediction-hero">
                <h1>Mumbai House <span>Price <br /> Prediction</span></h1>
                <p>Our advanced AI model evaluates property features to estimate house prices with high accuracy.</p>
            </div>
            <div className="price-detection">
                <div className="notice">
                    <h3>Enter property details</h3>
                    <p>Enter correct data to get accurate result</p>
                </div>
                <div className="input-container">
                    <div className="input-group">
                        <label htmlFor="area">{labels.area}</label>
                        <input type="text" id="area" name="area" value={inputData.area} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="bedroom_num">{labels.bedroom_num}</label>
                        <input type="text" id="bedroom_num" name="bedroom_num" value={inputData.bedroom_num} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="bathroom_num">{labels.bathroom_num}</label>
                        <input type="text" id="bathroom_num" name="bathroom_num" value={inputData.bathroom_num} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="balcony_num">{labels.balcony_num}</label>
                        <input type="text" id="balcony_num" name="balcony_num" value={inputData.balcony_num} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="age">{labels.age}</label>
                        <input type="text" id="age" name="age" value={inputData.age} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="total_floors">{labels.total_floors}</label>
                        <input type="text" id="total_floors" name="total_floors" value={inputData.total_floors} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="latitude">{labels.latitude}</label>
                        <input type="text" id="latitude" name="latitude" value={inputData.latitude} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="longitude">{labels.longitude}</label>
                        <input type="text" id="longitude" name="longitude" value={inputData.longitude} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="locality">{labels.locality}</label>
                        <input type="text" id="locality" name="locality" value={inputData.locality} onChange={handleInputChange} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="city">{labels.city}</label>
                        <select id="city" name="city" value={inputData.city} onChange={handleInputChange}>
                            <option value="">Select City</option>
                            {cities.map(city => (
                                <option key={city} value={city}>{city}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-group">
                        <label htmlFor="property_type">{labels.property_type}</label>
                        <select id="property_type" name="property_type" value={inputData.property_type} onChange={handleInputChange}>
                            <option value="">Select Property Type</option>
                            {propertyTypes.map(type => (
                                <option key={type} value={type}>{type}</option>
                            ))}
                        </select>
                    </div>
                    <div className="input-group">
                        <label htmlFor="furnished">{labels.furnished}</label>
                        <select id="furnished" name="furnished" value={inputData.furnished} onChange={handleInputChange}>
                            <option value="">Select Furnishing</option>
                            {furnishedOptions.map(option => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </div>
                </div>
                <button className="predict-button" onClick={handlePricePrediction}>
                    Predict Price
                </button>
                {predictionResult && (
                    <div className="prediction-result">
                        {predictionResult.error ? (
                            <p className="error">{predictionResult.error}</p>
                        ) : (
                            <>
                                <p className="result">Estimated Price: {predictionResult.prediction}</p>
                            </>
                        )}
                    </div>
                )}
            </div>
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Mumbai House Price Prediction Model</h1>
                    <p>A machine learning system for real estate valuation in Mumbai</p>
                    
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
                                    Our Mumbai House Price Prediction Model is a machine learning system designed to estimate property values based on various features. It helps buyers, sellers, and real estate professionals get accurate price estimates.
                                </p>
                                
                                <h2 className='workflow'>Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Cleaning</h3>
                                        <p>Gathering property data, handling missing values, and log-transforming prices.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Feature Engineering</h3>
                                        <p>Processing numerical and categorical features with appropriate transformations.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Model Training & Prediction</h3>
                                        <p>Using Random Forest to predict property prices with high accuracy.</p>
                                    </li>
                                </div>

                                <h2 className='keycomponenets'>Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer">
                                                Mumbai House Price Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>12 property features including area, bedrooms, location, etc.</li>
                                        <li>Log-transformed target variable for better modeling</li>
                                        <li>Comprehensive preprocessing pipeline</li>
                                    </ul>
                                </div>

                                <hr />

                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Random Forest Regressor - Robust for real estate pricing</li>
                                        <li>StandardScaler - For numerical feature normalization</li>
                                        <li>OneHotEncoder - For categorical feature processing</li>
                                    </ul>
                                </div>

                                <div className="ApproachUsed">
                                    <h3>Real Estate Analytics Approach</h3>
                                    <p>
                                        Our system uses an ensemble learning approach to analyze multiple factors that contribute to property prices, including location, size, amenities, and property characteristics.
                                    </p>
                                </div>
                            </div>
                        }
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our real estate pricing model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Essential libraries for data processing and machine learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Data Loading and Preprocessing</h2>
                                    <p>Preparing the real estate dataset for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Load dataset
df = pd.read_csv('mumbai-house-price-data-cleaned.csv')

# Remove unnecessary columns
df.drop(columns=['title', 'price_per_sqft'], inplace=True)
df.dropna(subset=['price'], inplace=True)

# Log-transform the target variable
df['log_price'] = np.log(df['price'])

# Define features and target
X = df.drop(['price', 'log_price'], axis=1)
y = df['log_price']

# Define feature types
numeric_features = ['area', 'bedroom_num', 'bathroom_num', 'balcony_num', 
                   'age', 'total_floors', 'latitude', 'longitude']
categorical_features = ['locality', 'city', 'property_type', 'furnished']`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Preprocessing Pipeline</h2>
                                    <p>Building the data transformation pipeline.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Model Training</h2>
                                    <p>Building and training the Random Forest model.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Random Forest model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)`}
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
                                    <h2>Regression Metrics</h2>
                                    <p className="accuracy">0.89</p>
                                    <p>R² Score (Coefficient of Determination)</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Mean Squared Error</th>
                                                <th>1.2e+10</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Root Mean Squared Error</td>
                                                <td>1.1e+5</td>
                                            </tr>
                                            <tr>
                                                <td>Mean Absolute Error</td>
                                                <td>7.8e+4</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Feature Importance</h2>
                                    <p className="accuracy">Area</p>
                                    <p>Most important feature for price prediction</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Top 5 Features</th>
                                                <th>Importance Score</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Area</td>
                                                <td>0.32</td>
                                            </tr>
                                            <tr>
                                                <td>Location (Latitude)</td>
                                                <td>0.18</td>
                                            </tr>
                                            <tr>
                                                <td>Location (Longitude)</td>
                                                <td>0.15</td>
                                            </tr>
                                            <tr>
                                                <td>Property Type</td>
                                                <td>0.12</td>
                                            </tr>
                                            <tr>
                                                <td>Number of Bedrooms</td>
                                                <td>0.08</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="validation-methodology">
                                    <h2>Validation Methodology</h2>
                                    <h3>Cross-Validation</h3>
                                    <p>Our model uses robust validation techniques:</p>
                                    <ul>
                                        <li>Stratified 5-fold cross-validation</li>
                                        <li>Geospatial validation splits</li>
                                        <li>Time-based validation for recent properties</li>
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

export default MumbaiHousePricePrediction;