import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Home from "./pages/Home/Home";
import Model from "./pages/Models/Models";
import Stock_prediction from "./pages/Stock Prediction Model/Stock_prediction";
import Spam from './pages/Spam detection/Spam_detection'
import Movie from './pages/Movie Recommendation/Movie_recommendation'
import Housing from './pages/Housing price prediction/hpp'
import Heart_disease from './pages/Heart disease/Heart_disease'
import BreastCancer from "./pages/Breast Cancer/BreastCancer";
import SalesPrediction from './pages/sales prediction/salesPrediction';
import CarPricePrediction from './pages/Car Price Prediction/CarPricePrediction'
import BankChurnPrediction from "./pages/BankChern/BankChernPrediction";
import EmployeeAttritionPrediction from "./pages/Employeeattrition/EmployeeAttritionPrediction";
import DiabetesPrediction from "./pages/Diabetes Prediction Model/DiabetesPrediction";

const App = () => {


  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<Model />} />
        <Route path='/stock_prediction' element={<Stock_prediction/>} />
        <Route path='/spam' element={<Spam/>} />
        <Route path='/movie' element={<Movie/>} />
        <Route path='/hpp' element={<Housing/>} />
        <Route path='/heart' element={<Heart_disease/>} />
        <Route path='/BreastCancer' element={<BreastCancer/>} />
        <Route path='/sales_prediction' element={<SalesPrediction />} />
        <Route path='/car_price_prediction' element={<CarPricePrediction />} />
        <Route path='/bank_churn' element={<BankChurnPrediction />} />
        <Route path='/employee_attrition' element={<EmployeeAttritionPrediction />} />
        <Route path='/diabetes_prediction' element={<DiabetesPrediction />} />
      </Routes>
    </Router>
  );
};

export default App;
