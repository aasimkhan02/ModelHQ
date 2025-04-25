import React, { useState } from 'react';
import { Link, useLocation } from "react-router-dom";
import './Navbar.css';

const Navbar = () => {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const location = useLocation();
    
    if (location.pathname === '/stock_prediction') {
        return null;
    }
    if (location.pathname === '/spam_detection') {
        return null;
    }
    if (location.pathname === '/hpp') {
        return null;
    }
    if (location.pathname === '/heart') {
        return null;
    }
    if (location.pathname === '/diabetes') {
        return null;
    }
    if (location.pathname === '/BreastCancer') {
        return null;
    }
    if (location.pathname === '/sales_prediction') {
        return null;
    }
    if (location.pathname === '/car_price_prediction') {
        return null;
    }
    if (location.pathname === '/bank_churn') {
        return null;
    }
    if (location.pathname === '/employee_attrition') {
        return null;
    }
    if (location.pathname === '/diabetes_prediction') {
        return null;
    }

    return (
        <div className='Navbar'>
            <h2 className='logo'>ModelHub</h2>
            <ul className='nav-list'>
                <li><Link to="/" className="nav-link"><h3>Home</h3></Link></li>
                <li
                    className='features-dropdown'
                    onMouseEnter={() => setDropdownOpen(true)}
                    onMouseLeave={() => setDropdownOpen(false)}
                >
                    <h3>Features</h3>
                    {dropdownOpen && (
                        <ul className="dropdown-menu">
                            <li><Link to="/models" className="nav-link">Models</Link></li>
                            <li><Link to="/learning" className="nav-link">Tutorials</Link></li>
                        </ul>
                    )}
                </li>
                <li><Link to="/contact" className="nav-link"><h3>Contact</h3></Link></li>
            </ul>
            <button className='Signup'><h3>Signup</h3></button>
        </div>
    );
}

export default Navbar;
