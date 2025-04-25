import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Home from "./pages/Home/Home";
import Model from "./pages/Models/Models";
import Stock_prediction from "./pages/Stock Prediction Model/Stock_prediction";
import SpamDetection from "./pages/Spam detection/Spam_detection";
import Heart_disease from './pages/Heart disease/Heart_disease'
import BreastCancer from "./pages/Breast Cancer/BreastCancer";
import SalesPrediction from './pages/sales prediction/salesPrediction';
import CarPricePrediction from './pages/Car Price Prediction/CarPricePrediction'
import BankChurnPrediction from "./pages/BankChern/BankChurnPrediction";
import EmployeeAttritionPrediction from "./pages/Employeeattrition/EmployeeAttritionPrediction";
import DiabetesPrediction from "./pages/Diabetes Prediction Model/DiabetesPrediction";
import HousingPricing from './pages/Housing price prediction/HousingPrice'

const App = () => {


  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<Model />} />
        <Route path='/stock_prediction' element={<Stock_prediction/>} />
        <Route path='/spam_detection' element={<SpamDetection/>} />
        <Route path='/heart' element={<Heart_disease/>} />
        <Route path='/BreastCancer' element={<BreastCancer/>} />
        <Route path='/sales_prediction' element={<SalesPrediction />} />
        <Route path='/car_price_prediction' element={<CarPricePrediction />} />
        <Route path='/bank_churn' element={<BankChurnPrediction />} />
        <Route path='/employee_attrition' element={<EmployeeAttritionPrediction />} />
        <Route path='/diabetes_prediction' element={<DiabetesPrediction />} />
        <Route path='/hpp' element={<HousingPricing />} />
      </Routes>
    </Router>
  );
};

export default App;
