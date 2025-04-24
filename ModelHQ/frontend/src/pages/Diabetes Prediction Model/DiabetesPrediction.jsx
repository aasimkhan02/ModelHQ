import React, { useState } from 'react';
import './DiabetesPrediction.css';

const DiabetesPrediction = () => {
    const [inputData, setInputData] = useState({
        Pregnancies: '',
        Glucose: '',
        BloodPressure: '',
        SkinThickness: '',
        Insulin: '',
        BMI: '',
        DiabetesPedigreeFunction: '',
        Age: ''
    });
    const [predictionResult, setPredictionResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setInputData({ ...inputData, [name]: value });
    };

    const handlePrediction = async () => {
        // Validate input data
        for (const key in inputData) {
            if (inputData[key] === '' || inputData[key] === null) {
                alert(`Please fill in the '${key}' field.`);
                return;
            }
        }
    
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/predict/diabetes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });
    
            const data = await response.json();
            if (data.status === 'success') {
                setPredictionResult(data);
            } else {
                throw new Error(data.message || 'Prediction failed');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            setPredictionResult(null);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="DiabetesPrediction">
            <h1>Diabetes Prediction</h1>
            <div className="input-container">
                {Object.keys(inputData).map((key) => (
                    <div className="input-group" key={key}>
                        <label htmlFor={key}>{key}</label>
                        <input
                            type="number"
                            id={key}
                            name={key}
                            value={inputData[key]}
                            onChange={handleInputChange}
                        />
                    </div>
                ))}
            </div>
            <button className="predict-button" onClick={handlePrediction} disabled={isLoading}>
                {isLoading ? 'Predicting...' : 'Predict'}
            </button>
            {predictionResult && (
                <div className="prediction-result">
                    <p><strong>Prediction:</strong> {predictionResult.prediction === 1 ? 'Diabetic' : 'Non-Diabetic'}</p>
                    <p><strong>Probability:</strong> {predictionResult.probability.toFixed(2)}</p>
                </div>
            )}
        </div>
    );
};

export default DiabetesPrediction;