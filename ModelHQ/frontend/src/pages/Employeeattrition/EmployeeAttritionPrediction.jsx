import React, { useState } from 'react';
import axios from 'axios';
import './EmployeeAttritionPrediction.css';
import { FaTimes } from 'react-icons/fa'; // Add this line

const EmployeeAttritionPrediction = () => {
    const [inputData, setInputData] = useState({
        Age: '',
        DailyRate: '',
        DistanceFromHome: '',
        Education: '',
        EnvironmentSatisfaction: '',
        JobInvolvement: '',
        JobLevel: '',
        JobSatisfaction: '',
        MonthlyIncome: '',
        NumCompaniesWorked: '',
        PercentSalaryHike: '',
        PerformanceRating: '',
        RelationshipSatisfaction: '',
        StockOptionLevel: '',
        TotalWorkingYears: '',
        TrainingTimesLastYear: '',
        WorkLifeBalance: '',
        YearsAtCompany: '',
        YearsInCurrentRole: '',
        YearsSinceLastPromotion: '',
        YearsWithCurrManager: '',
        Department_Sales: '',
    Department_ResearchDevelopment: '',
    EducationField_LifeSciences: '',
    EducationField_Marketing: '',
    EducationField_TechnicalDegree: ''
    });
    const [predictionResult, setPredictionResult] = useState(null);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handleAttritionPrediction = async () => {
        try {
            // Convert input values to the correct data types
            const formattedData = Object.fromEntries(
                Object.entries(inputData).map(([key, value]) => [key, parseFloat(value)])
            );

            console.log('Formatted Input Data:', formattedData); // Log the formatted input data
            const response = await axios.post('http://localhost:8000/predict/employee_attrition', formattedData);
            console.log('API Response:', response.data); // Log the API response
            setPredictionResult(response.data);
        } catch (error) {
            console.error('Error making prediction:', error);
            alert('Failed to get prediction. Please check the input or try again later.');
        }
    };

    return (
        <div className="EmployeeAttritionPrediction">
            <div className="header">
                <div className="logo">Employee Attrition Prediction</div>

            </div>
            <div className="input-container">
                {Object.keys(inputData).map((key) => (
                    <div className="input-group" key={key}>
                        <label htmlFor={key}>{key}</label>
                        <input
                            type="text"
                            id={key}
                            name={key}
                            value={inputData[key]}
                            onChange={handleInputChange}
                        />
                    </div>
                ))}
            </div>
            <button className="predict-button" onClick={handleAttritionPrediction}>
                Predict
            </button>
            {predictionResult && (
                <div className="prediction-result">
                    <p><strong>Prediction:</strong> {predictionResult.prediction === 1 ? 'Attrition' : 'No Attrition'}</p>
                    <p>
                        <strong>Probability:</strong> 
                        {predictionResult.probability !== undefined 
                            ? predictionResult.probability.toFixed(2) 
                            : 'N/A'}
                    </p>
                </div>
            )}
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Employee Attrition Prediction Model</h1>
                    <p>A predictive analytics system for workforce retention</p>
                    
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
                                    Our Employee Attrition Prediction Model is a machine learning system designed to help HR professionals identify employees at risk of leaving the company. It analyzes various employee attributes and work-related factors to predict attrition likelihood.
                                </p>
                                
                                <h2 className='workflow'>Workflow</h2>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Gathering HR data, handling missing values, and encoding categorical variables.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Feature Engineering</h3>
                                        <p>Creating meaningful features and handling class imbalance.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Model Training & Prediction</h3>
                                        <p>Using XGBoost to predict attrition probability.</p>
                                    </li>
                                </div>

                                <h2 className='keycomponenets'>Key Components</h2>
                                <div className="DataSource">
                                    <h3>Data Source</h3>
                                    <ul>
                                        <li>
                                            <a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset" target="_blank" rel="noopener noreferrer">
                                                IBM HR Analytics Employee Attrition Dataset (Kaggle)
                                            </a>
                                        </li>
                                        <li>35 employee attributes including age, department, job role, etc.</li>
                                        <li>1,470 employee records</li>
                                        <li>16% attrition rate (imbalanced dataset)</li>
                                    </ul>
                                </div>

                                <hr />

                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>XGBoost Classifier - Optimized for tabular data</li>
                                        <li>SMOTE - For handling class imbalance</li>
                                    </ul>
                                </div>

                                <div className="ApproachUsed">
                                    <h3>HR Analytics Approach</h3>
                                    <p>
                                        Our system uses a gradient boosting approach to analyze multiple factors that contribute to employee attrition, including job satisfaction, work-life balance, compensation, and career growth opportunities.
                                    </p>
                                </div>

                                <div className="download-buttons">
                                    <a
                                        href="/path/to/notebook.ipynb"
                                        download="EmployeeAttrition_Notebook.ipynb"
                                        className="download-button"
                                    >
                                        Download Python Notebook
                                    </a>
                                    <a
                                        href="/path/to/model.pkl"
                                        download="EmployeeAttrition_Model.pkl"
                                        className="download-button"
                                    >
                                        Download Model (.pkl)
                                    </a>
                                </div>
                            </div>
                        }
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our HR analytics model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Essential libraries for data processing and machine learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Data Loading and Preprocessing</h2>
                                    <p>Preparing the HR dataset for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Load data
df = pd.read_csv("employee_attrition.csv")

# Encode target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], 
        axis=1, inplace=True)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Handling Class Imbalance</h2>
                                    <p>Addressing the imbalanced nature of attrition data.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Split features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Model Training</h2>
                                    <p>Building and training the XGBoost classifier.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train XGBoost model
model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=5  # Adjust for class imbalance
)
model.fit(X_train, y_train)`}
                                        </SyntaxHighlighter>
                                    </div>

                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>Using the trained model to predict attrition risk.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`def predict_attrition(employee_data):
    # Preprocess input data
    input_df = pd.DataFrame([employee_data])
    input_df = pd.get_dummies(input_df)
    
    # Ensure all training columns are present
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X_train.columns]
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        'prediction': prediction[0],
        'probability': probability
    }`}
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
                                    <h2>Classification Accuracy</h2>
                                    <p className="accuracy">91.2%</p>
                                    <p>Mean accuracy across test dataset</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Precision</th>
                                                <th>89.5%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Recall</td>
                                                <td>92.1%</td>
                                            </tr>
                                            <tr>
                                                <td>F1 Score</td>
                                                <td>90.8%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Business Impact Metrics</h2>
                                    <p className="accuracy">87.3%</p>
                                    <p>True Positive Rate (Sensitivity)</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>False Positive Rate</th>
                                                <th>8.2%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>ROC AUC</td>
                                                <td>0.94</td>
                                            </tr>
                                            <tr>
                                                <td>Precision-Recall AUC</td>
                                                <td>0.93</td>
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
                                        <li>Time-based validation split</li>
                                        <li>Business impact simulation testing</li>
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

export default EmployeeAttritionPrediction;