import React from 'react'
import "./Home.css"
import Robot from './../../assets/robot.png'
import Marquee from "react-fast-marquee";
import finance from "./../../assets/finance.png"
import code from './../../assets/code.png'
import singlebook from './../../assets/singlebook.png'
import laptop from './../../assets/laptop.png'
import Building from './../../assets/building.jpg'
import heart from './../../assets/disease.png'

const industries = [
    "Healthcare",
    "Finance",
    "Education",
    "Retail",
    "Manufacturing",
    "Transportation",
    "Agriculture",
    "Energy",
    "Telecommunications",
    "Real Estate"
];


const Home = () => {
    return (
        <div className='Home'>
            <div className="Hero">
                <hr className='hr1'/>
                <hr className='hr2'/>
                <hr className='hr3'/>
                <hr className='hr4'/>
                <div className="hero-texts">
                    <h3>Redefining Predictions</h3>
                    <h1>ModelHub</h1>
                    <h2>Bringing Every Prediction Model to Your Fingertips</h2>
                    <button>Explore Models</button>
                </div>
                <img src={Robot} alt="" />
            </div>
            <Marquee className='marquee' speed={100} gradient={false}>
                {industries.map((industry, index) => (
                    <li key={index} style={{ margin: "0 30px", listStyle: "none", fontSize: "25px", fontWeight: "500", letterSpacing: '1px' }}>
                        {industry}
                    </li>
                ))}
            </Marquee>
            <div className="Features">
                <h1>FEATURES</h1>
                <h2>
                    ModelHQ is a platform where you can 
                    access and use multiple prediction 
                    models in one place, making real-time 
                    predictions simple and accessible. Our 
                    goal is to simplify machine learning by 
                    allowing users to explore, interact with, 
                    and compare models effortlessly—no 
                    complex setups or coding required.
                </h2>
                <div className="feature-cards">
                    <li>
                        <h1>Multiple Prediction Models</h1>
                        <img src={finance} alt="" />
                    </li>
                    <li>
                        <h1>Access to Model Code</h1>
                        <img src={code} alt="" className='codeimg'/>
                    </li>
                    <li>
                        <h1>Access to Model Code</h1>
                        <img src={singlebook} alt="" className='singlebookimg'/>
                    </li>
                    <li>
                        <h1>Access to Model Code</h1>
                        <img src={laptop} alt="" className='laptopimg'/>
                    </li>
                </div>
            </div>
            <div className="Model-categories">
                <h1>Model Categories</h1>
                <div className="categories">
                    <li className='Category-heading'>
                        <h2>Discover Powerful Prediction Models Across Industries</h2>
                        <button>Discover</button>
                    </li>
                    <li className='Healthcare'>
                        <h2>Healthcare</h2>
                    </li>
                    <li className='Buisness'>
                        <h2>Business</h2>
                    </li>
                    <li className='Retail'>
                        <h2>Retail</h2>
                    </li>
                    <li className='Education'>
                        <h2>Education</h2>
                    </li>
                </div>
            </div>
            <div className="Why-us">
                <h1 className='Heading'>WHY US</h1>
                <div className="Whyus-container">
                    <img src={Building} alt="" />
                    <div className="Whyus-content">
                        <h2>Unlock the Power of Prediction Models, All in One Place</h2>
                        <h3>ModelHQ simplifies machine learning by giving you access to diverse prediction models, real-time interaction, full code transparency, and comprehensive tutorials—all designed to help you learn, experiment, and innovate effortlessly</h3>
                        <button>Contact Us</button>
                    </div>
                </div>
            </div>
            <div className="popular-models">
                <h1 className='Heading'>Popular Models</h1>
                <div className="first-model model">
                    <div className="model-header">
                        <img src={heart} alt="" />
                        <h2>Heart Disease Prediction Model</h2>
                    </div>
                    <h3>Predict the likelihood of heart disease based on key health indicators, helping in early detection and risk assessment.</h3>
                    <h2 className='buildwith'>Built with: Linear Regression</h2>
                    <button>Test Now</button>
                </div>
                <div className="second-model model">
                    <div className="model-header">
                        <img src={heart} alt="" />
                        <h2>Diabetes Prediction Model</h2>
                    </div>
                    <h3>Assess the risk of developing diabetes using key health metrics for early intervention.</h3>
                    <h2 className='buildwith'>Built with: Logistic Regression</h2>
                    <button>Test Now</button>
                </div>
            </div>
        </div>
    )
}

export default Home
