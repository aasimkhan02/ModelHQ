import React, { useState } from 'react';
import './Heart_disease.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const HeartDiseasePrediction = () => {
    const [inputData, setInputData] = useState({
        age: '',
        sex: '',
        cp: '',
        trestbps: '',
        chol: '',
        fbs: '',
        restecg: '',
        thalach: '',
        exang: '',
        oldpeak: '',
        slope: '',
        ca: '',
        thal: ''
    });
    const [activeSection, setActiveSection] = useState('overview');
    const [openSection, setOpenSection] = useState(null);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [predictionResult, setPredictionResult] = useState(null);


    const toggleSection = (section) => {
        setOpenSection(openSection === section ? null : section);
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleHeartDiseasePrediction = async () => {
        try {
            // Validate input data
            const requiredFields = [
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ];
            for (const field of requiredFields) {
                if (!inputData[field]) {
                    setPredictionResult({ error: `Field "${field}" is required.` });
                    return;
                }
            }
    
            const response = await fetch('http://localhost:8000/predict/heart_disease', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    age: parseFloat(inputData.age),
                    sex: parseInt(inputData.sex),
                    cp: parseInt(inputData.cp),
                    trestbps: parseFloat(inputData.trestbps),
                    chol: parseFloat(inputData.chol),
                    fbs: parseInt(inputData.fbs),
                    restecg: parseInt(inputData.restecg),
                    thalach: parseFloat(inputData.thalach),
                    exang: parseInt(inputData.exang),
                    oldpeak: parseFloat(inputData.oldpeak),
                    slope: parseInt(inputData.slope),
                    ca: parseInt(inputData.ca),
                    thal: parseInt(inputData.thal)
                })
            });
    
            const result = await response.json();
            if (result.status === 'success') {
                setPredictionResult({
                    prediction: result.prediction === 1 ? 'Heart Disease Detected' : 'No Heart Disease',
                    probability: (result.probability * 100).toFixed(2) + '%'
                });
            } else {
                setPredictionResult({ error: result.message });
            }
        } catch (error) {
            setPredictionResult({ error: 'An error occurred while making the prediction.' });
        }
    };

    const labels = {
        age: "Age",
        sex: "Sex",
        cp: "Chest Pain Type",
        trestbps: "Resting Blood Pressure",
        chol: "Serum Cholesterol",
        fbs: "Fasting Blood Sugar",
        restecg: "Resting Electrocardiographic Results",
        thalach: "Maximum Heart Rate Achieved",
        exang: "Exercise Induced Angina",
        oldpeak: "ST Depression Induced by Exercise",
        slope: "Slope of the Peak Exercise ST Segment",
        ca: "Number of Major Vessels Colored by Fluoroscopy",
        thal: "Thalassemia"
    };

    return (
        <div className="HeartDiseasePrediction">
            <div className="header">
                <div className="logo">ModelHub</div>
                <FaAtlas className="book-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="HeartDiseasePrediction-hero">
                <h1>Heart Disease <span>Prediction <br /> Model</span></h1>
                <p>Our advanced AI model evaluates key health indicators to assess 
                your heart disease risk with high accuracy.</p>
            </div>
            <div className="heart-detection">
                <div className="notice">
                    <h3>Enter health parameters</h3>
                    <p>Enter correct data to get accurate result</p>
                </div>
                <div className="input-container">
                    {Object.keys(inputData).map((key) => (
                        <div className="input-group" key={key}>
                            <label htmlFor={key}>{labels[key]}</label>
                            <input
                                type="text"
                                id={key}
                                name={key}
                                placeholder={`Enter ${key}`}
                                value={inputData[key]}
                                onChange={handleInputChange}
                            />
                        </div>
                    ))}
                </div>
                <button className="predict-button" onClick={handleHeartDiseasePrediction}>
                    Predict
                </button>
                {predictionResult && (
                    <div className="prediction-result">
                        {predictionResult.error ? (
                            <p className="error">{predictionResult.error}</p>
                        ) : (
                            <>
                                <p className="result">Prediction: {predictionResult.prediction}</p>
                                <p className="probability">Probability: {predictionResult.probability}</p>
                            </>
                        )}
                    </div>
                )}
            </div>
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Heart Disease Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning heart disease prediction system</p>
                    
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
                                    Our Heart Disease Prediction Model is a machine learning system designed to assess the risk of heart disease based on key health indicators. It provides predictions with high accuracy to assist healthcare professionals.
                                </p>
                                
                                <h2 className='workflow'>Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Gathering patient data, cleaning, and normalizing for model training.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>Using Logistic Regression for binary classification of heart disease.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Providing predictions with probability scores and accuracy metrics.</p>
                                    </li>
                                </div>

                                <h2 className='keycomponenets'>Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/ronitf/heart-disease-uci" target="_blank" rel="noopener noreferrer">
                                                UCI Heart Disease Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>Features – Age, Cholesterol, Blood Pressure, etc.</li>
                                        <li>Data preprocessing – Handling missing values and scaling features.</li>
                                    </ul>
                                </div>

                                <hr />

                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Logistic Regression – Used for binary classification of heart disease.</li>
                                    </ul>
                                </div>

                                <div className="ApproachUsed">
                                    <h3>Approach</h3>
                                    <p>
                                        Our system leverages Logistic Regression to predict heart disease based on features like age, cholesterol, and blood pressure. It provides interpretable results for clinical use.
                                    </p>
                                </div>
                                <div className="download-buttons">
                                    <a
                                        href="./../../../../backend/models/Heart disease/Heart_Disease.ipynb"
                                        download="HeartDisease.ipynb"
                                        className="download-button"
                                    >
                                        Download Python Notebook
                                    </a>
                                    <a
                                        href="./../../../../backend/models/Heart disease/heart_disease_model.h5"
                                        download="HeartDisease.pkl"
                                        className="download-button"
                                    >
                                        Download Model File
                                    </a>
                                </div>
                            </div>}
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Essential libraries for data manipulation and model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>numpy, pandas: For handling and processing heart disease data.</li>
                                        <li>sklearn.model_selection.train_test_split: Splits data into training and testing sets.</li>
                                        <li>sklearn.linear_model.LogisticRegression: Used to build the heart disease prediction model.</li>
                                        <li>sklearn.metrics.accuracy_score: Evaluates model accuracy.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Data</h2>
                                    <p>We load the heart disease data and preprocess it for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Loading the data from the CSV file to a pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# Separating the data and labels
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train the Logistic Regression model using the training data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>}
                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>

                                <section className="metric-section">
                                    <h2>Accuracy</h2>
                                    <p className="accuracy">85.0%</p>
                                    <p>Mean accuracy across all tested data</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Training Accuracy</th>
                                                <th>86.0%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Testing Accuracy</td>
                                                <td>85.0%</td>
                                            </tr>
                                            <tr>
                                                <td>Cross-Validation Accuracy</td>
                                                <td>84.5%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Precision and Recall</h2>
                                    <p className="accuracy">84.0%</p>
                                    <p>Correctly identified heart disease and healthy cases</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Precision</th>
                                                <th>0.84</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Recall</td>
                                                <td>0.83</td>
                                            </tr>
                                            <tr>
                                                <td>F1 Score</td>
                                                <td>0.835</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="validation-methodology">
                                    <h2>Validation Methodology</h2>
                                    <h3>Cross-Validation</h3>
                                    <p>Our model uses cross-validation to ensure robust performance. This approach:</p>
                                    <ul>
                                        <li>Splits the dataset into multiple folds</li>
                                        <li>Trains and tests the model on different folds</li>
                                        <li>Aggregates results to evaluate overall performance</li>
                                    </ul>
                                </section>
                            </div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HeartDiseasePrediction;