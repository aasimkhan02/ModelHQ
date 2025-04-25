from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import xgboost as xgb
import joblib  # For loading the scaler
import pandas as pd
import cv2
import pickle
from tensorflow.keras.models import load_model

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
breast_cancer_model = tf.keras.models.load_model("./models/Breast Cancer/Breast_cancer_new.h5")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("./models/Heart disease/heart_disease_model.h5")
scaler = joblib.load("./models/Heart disease/scaler.pkl")

# Define input schemas
class StockInput(BaseModel):
    stock_symbol: str
    days: int = 10

class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "ML Model API is running!"}

@app.post("/predict/breast_cancer")
async def predict_breast_cancer(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = breast_cancer_model.predict(img)[0][0]
    predicted_class = "Malignant" if prediction > 0.5 else "Benign"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    
    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "model": "breast_cancer"
    }

@app.post("/predict/stock")
def predict_stock(data: StockInput):
    try:
        # Extract stock symbol and days from the input
        stock_symbol = data.stock_symbol
        days = data.days

        # Construct the model path dynamically
        model_path = f"./models/stock prediction/{stock_symbol}.h5"

        # Check if the model file exists
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Model for {stock_symbol} not found."
            }

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Fetch stock data using yfinance
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if stock_data.empty or len(stock_data) < 100:
            return {
                "status": "error",
                "message": f"Insufficient historical data for {stock_symbol}."
            }

        # Preprocess data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data[['Close']])
        last_100_days = scaled_data[-100:].reshape(1, 100, 1)

        # Generate predictions
        predicted_prices = []
        current_sequence = last_100_days.copy()

        for _ in range(days):
            pred = model.predict(current_sequence, verbose=0)[0][0]
            actual_price = float(scaler.inverse_transform([[pred]])[0][0])
            predicted_prices.append(actual_price)
            current_sequence = np.append(current_sequence[:, 1:, :], [[[pred]]], axis=1)

        return {
            "status": "success",
            "stock": stock_symbol,
            "predictions": predicted_prices
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/bank_churn")
def predict_bank_churn(data: dict):
    try:
        # Load the trained model
        with open('./models/Bank Churn/bank_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the scaler
        scaler = joblib.load('./models/Bank Churn/scaler.pkl')

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Scale numerical features
        numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),  # 1: Churn, 0: No Churn
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    try:
        print("Received data:", data.dict())  # Log the input data

        # Convert input data to a DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical features to match training
        input_df = pd.get_dummies(input_df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

        # Load reference columns from training (X_train.columns saved during preprocessing)
        reference_cols = joblib.load("./models/Heart disease/reference_columns.pkl")

        # Add missing columns with 0
        for col in reference_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training order
        input_df = input_df[reference_cols]

        # Scale numerical features
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make prediction
        prediction = xgb_model.predict(input_df)[0]
        probability = xgb_model.predict_proba(input_df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),  # 1: Heart Disease, 0: No Heart Disease
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/walmart_sales")
def predict_walmart_sales(data: dict):
    try:
        # Load the trained model
        model = xgb.Booster()
        model.load_model("./models/sales prediction/walmart_sales_model.h5")

        # Load the scaler
        scaler = joblib.load("./models/sales prediction/scaler.pkl")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all expected columns exist and are of correct type
        input_df = input_df.astype({
            "Store": int,
            "Temperature": float,
            "Fuel_Price": float,
            "CPI": float,
            "Unemployment": float,
            "Year": int,
            "WeekOfYear": int,
            "Store_Size_Category_Medium": int,
            "Store_Size_Category_Large": int,
            "IsHoliday_1": int
        })

        # Scale numerical features
        numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Convert to DMatrix
        dinput = xgb.DMatrix(input_df)

        # Make prediction
        prediction = model.predict(dinput)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/car_price")
def predict_car_price(data: dict):
    try:
        # Load the trained model
        model = joblib.load("./models/Car Price Prediction/car_price_model.pkl")

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
@app.post("/predict/employee_attrition")
def predict_employee_attrition(data: dict):
    try:
        # Load the trained model
        with open('./models/Employee/employee_attrition_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the scaler
        scaler = joblib.load('./models/Employee/scaler.pkl')

        # Load reference columns from training (X_train.columns saved during preprocessing)
        reference_cols = joblib.load('./models/Employee/reference_columns.pkl')

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Add missing columns with 0
        for col in reference_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training order
        input_df = input_df[reference_cols]

        # Scale numerical features
        input_df = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),  # 1: Attrition, 0: No Attrition
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
@app.post("/predict/diabetes")
def predict_diabetes(data: dict):
    try:
        # Validate input data
        required_columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        for col in required_columns:
            if col not in data or data[col] == '' or data[col] is None:
                return {
                    "status": "error",
                    "message": f"Missing or invalid value for '{col}'"
                }

        # Load the trained model
        model = load_model("./models/Diabetes/diabetes_model.h5")

        # Load the scaler
        with open("./models/Diabetes/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Load reference columns
        with open("./models/Diabetes/reference.pkl", "rb") as f:
            reference = pickle.load(f)

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Ensure input columns match the training columns
        for col in reference["input_columns"]:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[reference["input_columns"]]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        probability = model.predict(input_scaled)[0][0]
        prediction = 1 if probability > 0.5 else 0

        return {
            "status": "success",
            "prediction": prediction,
            "probability": float(probability)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/housing_price")
def predict_housing_price(data: dict):
    try:
        # Load the trained model
        with open("./models/Housing/linear_regression.pkl", "rb") as f:
            linear_model = pickle.load(f)

        # Load the scaler
        with open("./models/Housing/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Load reference columns
        with open("./models/Housing/reference_columns.pkl", "rb") as f:
            reference_columns = pickle.load(f)

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Ensure input columns match the training columns
        for col in reference_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[reference_columns]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = linear_model.predict(input_scaled)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict/mumbai_house_price")
def predict_mumbai_house_price(data: dict):
    try:
        # Load the full model pipeline (includes preprocessor + regressor)
        with open("./models/HousePrice/mumbai_house_price_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load feature reference
        with open("./models/HousePrice/feature_reference.pkl", "rb") as f:
            reference = pickle.load(f)

        expected_cols = reference["all_features"]

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all expected columns are present in the input
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with default value 0

        # Reorder the input DataFrame columns to match the training data
        input_df = input_df[expected_cols]

        # Make prediction using the loaded model (which includes preprocessing)
        prediction = model.predict(input_df)[0]

        return {
            "status": "success",
            "prediction": float(prediction)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
class SpamInput(BaseModel):
    email_text: str

@app.post("/predict/spam_detection")
def predict_spam_detection(data: SpamInput):
    try:
        model_path = "./models/spam/spam_classifier_full.pkl"

        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Model file not found at {model_path}"
            }

        # Load model safely
        try:
            model_obj = joblib.load(model_path)
        except:
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)

        # Extract the model (pipeline)
        if isinstance(model_obj, dict):
            model = model_obj.get("model")
            if model is None:
                return {
                    "status": "error",
                    "message": "Model dictionary does not contain 'model' key."
                }
        else:
            model = model_obj

        # Clean input
        email_text = data.email_text.strip()
        if not email_text:
            return {
                "status": "error",
                "message": "Email text cannot be empty."
            }

        # Use the pipeline to predict
        prediction = model.predict([email_text])[0]
        confidence = model.predict_proba([email_text])[0][1]  # Spam class prob

        return {
            "status": "success",
            "prediction": "Spam" if prediction == 1 else "Ham",
            "confidence": round(float(confidence), 4)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)