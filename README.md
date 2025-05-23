# 🧠 ModelHQ

**ModelHQ** is a full-stack web application that provides access to various AI/ML prediction models across multiple domains like healthcare, finance, and more. Each model is not only interactive and usable with user input, but also comes with:

- 📥 Downloadable source code
- 🔍 Line-by-line code explanation
- 📚 Research-backed implementations

> Built with **React (frontend)** and **FastAPI (backend)**, ModelHQ is designed as both a **usable tool** and an **educational platform**.

---

## 🚀 Features

- 🩺 **Healthcare**, 💰 **Finance**, and other domain-specific models
- ✅ Real-time predictions through a web interface
- 📘 Line-by-line code explanations for each model
- 📁 Downloadable model scripts and data
- 🧠 Models implemented based on research papers
- 🖥️ Clean, responsive frontend (React)
- ⚡ Fast backend API using FastAPI

---

## 🧪 Available Models

|-------------|--------------------------------|--------------------------------------------------|
| Domain      | Model Name                     | Description                                      |
|-------------|--------------------------------|--------------------------------------------------|
| Healthcare  | Breast Cancer Prediction       | Classifies tumors using medical features         |
| Healthcare  | Heart Disease Prediction       | Predicts heart disease risk using medical data   |
| Healthcare  | Diabetes Prediction            | Predicts diabetes likelihood                     |
| Finance     | Stock Price Prediction         | Forecasts stock prices using historical data     |
| Finance     | Bank Churn Prediction          | Predicts whether a customer will leave a bank    |
| Finance     | Sales Prediction               | Estimates future sales based on past data        |
| Finance     | Car Price Prediction           | Predicts resale price of used cars               |
| HR/Business | Employee Attrition Prediction  | Predicts if an employee is likely to leave       |
| Communication | Spam Detection               | Classifies text messages as spam or not          |
| Real Estate | House Price Prediction         | Predicts house prices based on various features  |
| More coming | ...                            | ...                                              |

> All models are backed by reputable research papers and adapted for production use.

---

## 🛠️ Getting Started

### 1. 📦 Clone the Repository

git clone https://github.com/aasimkhan02/ModelHQ.git
cd ModelHQ
2. 🐍 Backend Setup (FastAPI)
Works best with Python 3.8 or newer.

a. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
b. Install dependencies
pip install -r requirements.txt
c. 📁 Download the models folder
Download models folder from Google Drive
https://drive.google.com/drive/folders/1JdhL_8AVeUrqBPmaJcZZwPzFKRLirJ6D?usp=drive_link

Place the downloaded models folder into the backend directory:
# Example
ModelHQ/
├── backend/
│   └── models/   <-- Place it here
3. 💻 Frontend Setup (React)
cd frontend
npm install
npm run dev

4. 🔙 Run the Backend API
Go back to the root directory, activate your virtual environment (if not already):
cd backend
python uvicorn main:app --reload
The backend will be served at: http://localhost:8000

📂 Project Structure
ModelHQ/
├── backend/
│   ├── main.py
│   ├── models/                # Pre-trained model files
│   ├── utils/                 # Utilities, pre-processing
│   └── ...
├── frontend/
│   ├── src/
│   ├── public/
│   └── ...
├── requirements.txt
└── README.md
📌 Notes
All models are pre-trained, saving compute and making the platform lightweight.

If you’re facing any issues with large model files, check the models folder download and placement.

This is a public, open-source educational tool and not intended for production medical/financial decisions.

🤝 Contributing
Pull requests and suggestions are welcome! Feel free to open an issue or submit a PR.
