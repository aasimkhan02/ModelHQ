import React from 'react';
import './Models.css';
import { FaSearch, FaTrash, FaChartLine, FaBolt } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

const Models = () => {
    const navigate = useNavigate();

    return (
        <div className='Models'>
            <div className="Models-hero">
                <h1>Explore Powerful <span>Prediction<br />Models</span></h1>
                <h3>Discover and interact with state-of-the-art machine 
                    learning models. Find the perfect model for your 
                    use case with our comprehensive collection.
                </h3>
            </div>
            <div className="Multiple_models">
                <div className="search-container">
                    <div className="search-bar">
                        <FaSearch className="search-icon" />
                        <input type="text" placeholder="Search Model" className="search-input" />
                    </div>

                    <div className="dropdowns">
                        <select className="dropdown">
                            <option>Category</option>
                            <option>Healthcare</option>
                            <option>Finance</option>
                            <option>Education</option>
                        </select>
                        <select className="dropdown">
                            <option>Model Type</option>
                            <option>Regression</option>
                            <option>Classification</option>
                            <option>Neural Networks</option>
                        </select>
                        <button className="clear-btn"><FaTrash className="Trash-icon" /></button>
                    </div>
                </div>
                <div className="Trending-models">
                    <div className="tm-heading">
                        <FaChartLine className="chart-icon" />
                        <h1>Trending Models</h1>
                    </div>
                    <ul className="Model-list">
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Breast Cancer Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{color: '#007EA7'}}/>
                                    <h4 style={{fontWeight: 500}}>98%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI Powered Breast Cancer classifier using CNN</p>
                            <div className="Model-tags">
                                <li>Healthcare</li>
                                <li>CNN</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/BreastCancer')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Trending</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Stock Price Predictor</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{color: '#007EA7'}}/>
                                    <h4 style={{fontWeight: 500}}>92%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered model predicting future stock prices from market trends.</p>
                            <div className="Model-tags">
                                <li>Finance</li>
                                <li>Time-series</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/stock_prediction')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Car Price Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{ color: '#007EA7' }} />
                                    <h4 style={{ fontWeight: 500 }}>90%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered model predicting car prices based on features like mileage, year, and condition.</p>
                            <div className="Model-tags">
                                <li>Automotive</li>
                                <li>Regression</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/car_price_prediction')}>Try Now</button>
                        </li>
                    </ul>
                </div>
                <div className="Featured-models">
                    <div className="tm-heading">
                        <FaChartLine className="chart-icon" />
                        <h1>Featured Models</h1>
                    </div>
                    <ul className="Model-list">
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>House Price Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{color: '#007EA7'}}/>
                                    <h4 style={{fontWeight: 500}}>92%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered hpp using historical data</p>
                            <div className="Model-tags">
                                <li>Finance</li>
                                <li>Time-series</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/hpp')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Heart Disease detection</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{color: '#007EA7'}}/>
                                    <h4 style={{fontWeight: 500}}>98%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered Heart disease detection using patient data</p>
                            <div className="Model-tags">
                                <li>Healthcare</li>
                                <li>Regression</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/heart')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Diabetes Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{ color: '#007EA7' }} />
                                    <h4 style={{ fontWeight: 500 }}>98%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered diabetes prediction using patient data.</p>
                            <div className="Model-tags">
                                <li>Healthcare</li>
                                <li>Classification</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/diabetes_prediction')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Sales Prediction Model</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{ color: '#007EA7' }} />
                                    <h4 style={{ fontWeight: 500 }}>95%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered model predicting weekly sales for retail stores using historical data.</p>
                            <div className="Model-tags">
                                <li>Retail</li>
                                <li>Regression</li>
                                <li>Time-series</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/sales_prediction')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Trending</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Spam Detection <br />Model</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{color: '#007EA7'}}/>
                                    <h4 style={{fontWeight: 500}}>98%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered spam detection using historical data</p>
                            <div className="Model-tags">
                                <li>Spam</li>
                                <li>NLP</li>
                                <li>Classification</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/spam')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Bank Churn Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{ color: '#007EA7' }} />
                                    <h4 style={{ fontWeight: 500 }}>94%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered model predicting customer churn based on financial and demographic data.</p>
                            <div className="Model-tags">
                                <li>Finance</li>
                                <li>Classification</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/bank_churn')}>Try Now</button>
                        </li>
                        <li>
                            <div className="Marketing">
                                <p>Featured</p>
                            </div>
                            <div className="Model-heading">
                                <h2>Employee Attrition Prediction</h2>
                                <div className="Model-accuracy">
                                    <FaBolt className="bolt-icon" style={{ color: '#007EA7' }} />
                                    <h4 style={{ fontWeight: 500 }}>93%</h4>
                                </div>
                            </div>
                            <p className='Model-subheading'>AI-powered model predicting employee attrition based on data.</p>
                            <div className="Model-tags">
                                <li>HR</li>
                                <li>Classification</li>
                            </div>
                            <hr />
                            <button className='TryNow stock' onClick={() => navigate('/employee_attrition')}>Try Now</button>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default Models;
