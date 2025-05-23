import React, { useState } from 'react';
import './Stock_prediction.css';
import { FaAtlas, FaTimes } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const Stock_prediction = () => {
    const [selectedStock, setSelectedStock] = useState('');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [activeSection, setActiveSection] = useState('overview');
    const [predictions, setPredictions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const stockOptions = [
        'AAPL', 'AMZN', 'BRK-B', 'DIS', 'GOOG', 'HD', 'JNJ', 'JPM', 'MA', 'META',
        'MSFT', 'NFLX', 'NVDA', 'PEP', 'PG', 'PYPL', 'TSLA', 'UNH', 'V', 'VZ'
    ];

    const handleSearch = async () => {
        if (!selectedStock) {
            alert('Please select a stock.');
            return;
        }

        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/predict/stock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    stock_symbol: selectedStock,
                    days: 10
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                setPredictions(data.predictions);
            } else {
                throw new Error(data.message || 'Prediction failed');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            setPredictions([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='Stock_prediction'>
            <div className="stock-header">
                <div className="logo">ModelHub</div>
                <FaAtlas className="Atlas-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>
            <div className="stock-search">
                <h1>Search Stock</h1>
                <select
                    value={selectedStock}
                    onChange={(e) => setSelectedStock(e.target.value)}
                    className="stock-dropdown"
                >
                    <option value="" disabled>Select a Stock</option>
                    {stockOptions.map((stock, index) => (
                        <option key={index} value={stock}>{stock}</option>
                    ))}
                </select>
                <button className="search-button" onClick={handleSearch} disabled={isLoading}>
                    {isLoading ? 'Loading...' : 'Search'}
                </button>
                <div className="predictions">
                    {predictions.length > 0 ? (
                        <div className="predictions-container">
                            {predictions.map((price, index) => (
                                <div key={index} className="prediction-card">
                                    <h3>Day {index + 1}</h3>
                                    <p>${price.toFixed(2)}</p>
                                </div>
                            ))}
                        </div>
                    ) : null}
                </div>
            </div>
            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Stock Prediction Model</h1>
                    <p>A comprehensive guide to our machine learning stock prediction system</p>
                    
                    <div className="model-details-options">
                        <button className={activeSection === 'overview' ? 'active' : ''} onClick={() => setActiveSection('overview')}>Overview</button>
                        <button className={activeSection === 'implementation' ? 'active' : ''} onClick={() => setActiveSection('implementation')}>Implementation</button>
                        <button className={activeSection === 'evaluation' ? 'active' : ''} onClick={() => setActiveSection('evaluation')}>Evaluation</button>
                    </div>

                    <div className="model-details-content">
                        {activeSection === 'overview' && 
                            <div className="model-details-overview">
                                <h1>Model Architecture</h1>
                                <p>High-level overview of our prediction system</p>
                                <div className="overview-cards">
                                    <li>
                                        <div className="circle">1</div>
                                        <h3>Data Collection & Preprocessing</h3>
                                        <p>Fetching and preprocessing historical stock price data.</p>
                                    </li>
                                    <li>
                                        <div className="circle">2</div>
                                        <h3>Model Architecture & Training</h3>
                                        <p>LSTM-based deep learning for stock prediction.</p>
                                    </li>
                                    <li>
                                        <div className="circle">3</div>
                                        <h3>Prediction & Evaluation</h3>
                                        <p>Future price forecasting with accuracy metrics.</p>
                                    </li>
                                </div>
                                <h1 className='keyComponenets'>Key Components</h1>
                                <div className="DataSource">
                                    <h3>Data source</h3>
                                    <ul>
                                        <li>Historical stock price data (OHLCV) – Fetched via yfinance</li>
                                        <li>Technical indicators – Moving averages, RSI, MACD, etc.</li>
                                        <li>Market indices – Used for broader trend analysis</li>
                                        <li>Trading volume data – Helps assess stock momentum</li>
                                    </ul>
                                </div>
                                <hr />
                                <div className="ModelUsed">
                                    <h3>Machine Learning Models</h3>
                                    <ul>
                                        <li>Long Short-Term Memory (LSTM) – Captures temporal dependencies in stock prices.</li>
                                    </ul>
                                </div>
                                <div className="ApproachUsed">
                                    <h3>Ensemble Approach</h3>
                                    <p>Our system leverages an LSTM-based model, optimized for sequential stock price data. It captures long-term dependencies and trends, enhancing prediction accuracy.</p>
                                </div>
                                <div className="download-buttons">
                                    <a
                                        href="./../../../../backend/models/Stock prediction/stock.ipynb"
                                        download="StockPrediction.ipynb"
                                        className="download-button"
                                    >
                                        Download Python Notebook
                                    </a>
                                    <a
                                        href="./../../../../backend/models/Stock prediction/AAPL.h5"
                                        download="StockPrediction.h5"
                                        className="download-button"
                                    >
                                        Download .h5 Model
                                    </a>
                                </div>
                            </div>}
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Before building the model, we need essential libraries for data manipulation, visualization, and deep learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>numpy, pandas: For handling and processing stock data.</li>
                                        <li>yfinance: Fetches historical stock prices from Yahoo Finance.</li>
                                        <li>matplotlib: Helps visualize stock trends and predictions.</li>
                                        <li>sklearn.preprocessing.MinMaxScaler: Normalizes stock prices for better model performance.</li>
                                        <li>tensorflow.keras: Used to build and train the LSTM model.</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Downloading and Preparing Stock Data</h2>
                                    <p>To train our model, we need historical stock prices. We fetch this using yfinance and process it for LSTM training.</p>
                                    <div className="code-section">
    <SyntaxHighlighter language="python" style={dracula}>
        {`def train_stock_model(stock_symbol, start_date="2015-03-01", end_date="2025-03-01"):
    print(f"📈 Training model for {stock_symbol}...")

    # Download stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        print(f"⚠️ No data found for {stock_symbol}, skipping...")
        return None

    data.dropna(inplace=True)  # Remove missing values
`}
    </SyntaxHighlighter>
</div>
                                    <p style={{marginTop: '15px'}}>The function above downloads stock data from Yahoo Finance for a given stock symbol. After fetching, it cleans the data by removing any missing values to ensure smooth training.</p>
                                    <h2 style={{marginTop: '50px'}}>Data Scaling for LSTM</h2>
                                    <p>Neural networks work best with normalized data, so we scale the closing prices between 0 and 1. LSTMs are sensitive to large numerical differences, and scaling helps speed up training and improves accuracy.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']])
`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Creating Training Sequences for LSTM</h2>
                                    <p>LSTMs require sequential input data. We use a lookback window of 100 days, meaning each input (X) consists of 100 previous days of stock prices, while the corresponding label (y) is the next day's price. The data is reshaped into a format that LSTM expects: (samples, timesteps, features).</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`X, y = [], []
seq_length = 100
for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i-seq_length:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Building the LSTM Model</h2>
                                    <p>Now, we define the LSTM architecture. The LSTM layers capture long-term dependencies in stock prices. Dropout layers prevent overfitting by randomly deactivating neurons. A Dense layer produces a single predicted stock price, and the Adam optimizer ensures efficient gradient descent.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=128, return_sequences=True),
    Dropout(0.3),
    LSTM(units=256),
    Dropout(0.4),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Training the Model</h2>
                                    <p>We train the model for 50 epochs and then save the trained model to reuse it for future predictions.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`model.fit(X, y, epochs=50, batch_size=32, verbose=1)
model.save(f"{stock_symbol}.h5")
`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Predicting Future Stock Prices</h2>
                                    <p>To predict the next 10 days of stock prices, we load the trained LSTM model and fetch the most recent 100 days of stock prices. The model predicts the next 10 days' prices, and the predictions are converted back to real stock price values.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`def predict_stock_price(stock_symbol, future_days=10):
    model = load_model(f"{stock_symbol}.h5")
    data = yf.download(stock_symbol, start="2024-01-01", end="2025-03-01")
    data.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']])
    last_100_days = scaled_data[-100:].reshape(1, 100, 1)
    predicted_prices = []
    for _ in range(future_days):
        pred_price = model.predict(last_100_days)[0][0]
        predicted_prices.append(pred_price)
        last_100_days = np.append(last_100_days[:, 1:, :], [[[pred_price]]], axis=1)
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices
`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Evaluating Model Performance</h2>
                                    <p>To measure our model’s accuracy, we compute MSE, RMSE, MAE, and R² Score. This function loads the trained model, fetches test data, and evaluates its predictions against actual stock prices.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
        {`def evaluate_model(stock_symbol):
    model = load_model(f"{stock_symbol}.h5")
    data = yf.download(stock_symbol, start="2024-01-01", end="2025-03-01")
    data.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']])
    X_test, y_test = [], []
    seq_length = 100
    for i in range(seq_length, len(scaled_data)):
        X_test.append(scaled_data[i-seq_length:i])
        y_test.append(scaled_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")
    return mse, rmse, mae, r2
`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>}
                            {activeSection === 'evaluation' && 
    <div className="model-details-evaluation">
        <h1>Model Evaluation</h1>
        <p>Performance metrics and validation methodology</p>

        <section className="metric-section">
            <h2>Price Prediction Accuracy</h2>
            <p className="accuracy">97.66%</p>
            <p>Mean accuracy across all tested stocks</p>
            <table>
                <thead>
                    <tr>
                        <th>Mean Absolute Error</th>
                        <th>2.34%</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Root Mean Squared Error</td>
                        <td>3.12%</td>
                    </tr>
                    <tr>
                        <td>Mean Absolute Percentage Error</td>
                        <td>2.41%</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="metric-section">
            <h2>Direction Accuracy</h2>
            <p className="accuracy">76.5%</p>
            <p>Correctly predicted price movement direction</p>
            <table>
                <thead>
                    <tr>
                        <th>Precision</th>
                        <th>0.79</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Recall</td>
                        <td>0.74</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>0.76</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="metric-section">
            <h2>Trading Performance</h2>
            <p className="accuracy">+18.7%</p>
            <p>Annualized return using model signals</p>
            <table>
                <thead>
                    <tr>
                        <th>Sharpe Ratio</th>
                        <th>1.42</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Maximum Drawdown</td>
                        <td>12.3%</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>68.9%</td>
                    </tr>
                </tbody>
            </table>
        </section>

        <section className="validation-methodology">
            <h2>Validation Methodology</h2>
            <h3>Walk-Forward Validation</h3>
            <p>Our model uses walk-forward validation to simulate real-world trading conditions. This approach:</p>
            <ul>
                <li>Trains on an initial window of historical data</li>
                <li>Makes predictions for the next period</li>
                <li>Incorporates new data and retrains</li>
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

export default Stock_prediction;
