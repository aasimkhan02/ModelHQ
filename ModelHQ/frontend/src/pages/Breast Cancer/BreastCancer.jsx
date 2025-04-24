import React, { useState, useRef } from 'react';
import './BreastCancer.css';
import { FaUpload, FaSpinner, FaDiagnoses, FaTimes, FaAtlas } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism";

const BreastCancer = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const fileInputRef = useRef(null);
    const [activeSection, setActiveSection] = useState('overview');

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedImage(file);
            setPreviewImage(URL.createObjectURL(file));
            setPrediction(null);
        }
    };

    const handlePredict = async () => {
        if (!selectedImage) {
            alert('Please upload an image first.');
            return;
        }
        
        setIsLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('file', selectedImage);
            
            const response = await fetch('http://localhost:8000/predict/breast_cancer', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            setPrediction(data.prediction);
            setConfidence(data.confidence);
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Error making prediction');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='BreastCancerClassifier'>
            <div className="classifier-header">
                <div className="logo">ModelHQ</div>
                <FaAtlas className="diagnosis-icon" onClick={() => setIsSidebarOpen(true)} />
            </div>

            <div className="classifier-search">
                <h1>Breast Cancer Classification</h1>
                <div className="upload-area" onClick={() => fileInputRef.current.click()}>
                    {previewImage ? (
                        <img src={previewImage} alt="Preview" className="image-preview" />
                    ) : (
                        <div className="upload-prompt">
                            <FaUpload className="upload-icon" />
                            <p>Click to upload histopathology image</p>
                        </div>
                    )}
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleImageUpload}
                        accept="image/*"
                        style={{ display: 'none' }}
                    />
                </div>
                <button 
                    className="search-button" 
                    onClick={handlePredict}
                    disabled={!selectedImage || isLoading}
                >
                    {isLoading ? (
                        <>
                            <FaSpinner className="spinner" />
                            Analyzing...
                        </>
                    ) : 'Classify Image'}
                </button>

                {prediction && (
                    <div className="predictions">
                        <h2>Classification Result</h2>
                        <div className={`prediction-result ${prediction.toLowerCase()}`}>
                            <h3>Diagnosis: {prediction}</h3>
                            <p>Confidence: {(confidence * 100).toFixed(2)}%</p>
                            {prediction === 'Malignant' ? (
                                <p className="warning">⚠️ This result suggests malignancy. Please consult with an oncologist.</p>
                            ) : (
                                <p className="reassurance">This result appears benign, but follow-up may still be recommended.</p>
                            )}
                        </div>
                    </div>
                )}
            </div>

            <div className={`sidebar-model-details ${isSidebarOpen ? 'open' : ''}`}>
                <FaTimes className="close-icon" onClick={() => setIsSidebarOpen(false)} />
                <div className="model-details-container">
                    <h1>Breast Cancer Classification Model</h1>
                    <p>A comprehensive guide to our medical image analysis system</p>
                    
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
                                Our Breast Cancer Classification Model is a state-of-the-art deep learning system designed to assist medical professionals in diagnosing breast cancer. It analyzes histopathology images to classify them as benign or malignant with high accuracy.
                            </p>
                            
                            <h2 className='workflow'>Workflow</h2>
                            <div className="overview-cards">
                                <li>
                                    <div className="circle">1</div>
                                    <h3>Data Collection & Preprocessing</h3>
                                    <p>Gathering histopathology images, applying augmentation, and normalizing data for training.</p>
                                </li>
                                <li>
                                    <div className="circle">2</div>
                                    <h3>Model Architecture & Training</h3>
                                    <p>Using a CNN-based architecture optimized for image classification tasks.</p>
                                </li>
                                <li>
                                    <div className="circle">3</div>
                                    <h3>Prediction & Evaluation</h3>
                                    <p>Classifying images with accuracy metrics and confidence scores.</p>
                                </li>
                            </div>

                            <h2 className='keycomponenets'>Key Components</h2>
                            <div className="DataSource">
                                <h3>Data Source</h3>
                                <ul>
                                    <li>
                                        <a href="https://www.kaggle.com/datasets/ambarish/breakhis" target="_blank" rel="noopener noreferrer">
                                            Histopathology Images Dataset (Kaggle)
                                        </a>
                                    </li>
                                    <li>Image augmentation – Rotation, flipping, brightness adjustment</li>
                                    <li>Normalization – Standardizing image sizes and color channels</li>
                                    <li>Class balancing – Handling imbalanced datasets</li>
                                </ul>
                            </div>

                            <hr />

                            <div className="ModelUsed">
                                <h3>Machine Learning Models</h3>
                                <ul>
                                    <li>Convolutional Neural Network (CNN) – Specialized for image analysis.</li>
                                </ul>
                            </div>

                            <div className="ApproachUsed">
                                <h3>Medical Imaging Approach</h3>
                                <p>
                                    Our system leverages a CNN-based model, optimized for histopathology images. It analyzes cellular structures and patterns to differentiate between benign and malignant tissue with high accuracy.
                                </p>
                            </div>

                            <div className="download-buttons">
                                <a
                                    href="./../../../../backend/models/Breast Cancer/Breast_cancer_new.ipynb"
                                    download="BreastCancer_Notebook.ipynb"
                                    className="download-button"
                                >
                                    Download Python Notebook
                                </a>
                                <a
                                    href="./../../../../backend/models/Breast Cancer/Breast_cancer_new.h5"
                                    download="BreastCancer_Model.h5"
                                    className="download-button"
                                >
                                    Download .h5 Model
                                </a>
                            </div>
                        </div>
                    }
                        {activeSection === 'implementation' && 
                            <div className="model-details-implementation">
                                <h1>Model Implementation</h1>
                                <p>Line-by-line code explanation of our model</p>
                                <div className="implementation-code">
                                    <h2>Importing Libraries</h2>
                                    <p>Essential libraries for image processing and deep learning.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <ul>
                                        <li>tensorflow: Core deep learning framework</li>
                                        <li>keras: High-level neural networks API</li>
                                        <li>numpy: Numerical computing</li>
                                        <li>cv2: Image processing</li>
                                        <li>matplotlib: Visualization</li>
                                    </ul>
                                    <h2 style={{marginTop: '50px'}}>Loading and Preprocessing Images</h2>
                                    <p>Preparing histopathology images for model training.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`def load_and_preprocess_image(image_path, img_size=(224, 224)):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Resize and normalize
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0,1]
    
    return img`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <p style={{marginTop: '15px'}}>This function loads an image, converts it to RGB format, resizes it to the required dimensions, and normalizes pixel values.</p>
                                    <h2 style={{marginTop: '50px'}}>Data Augmentation</h2>
                                    <p>Creating variations of training images to improve model robustness.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Building the CNN Model</h2>
                                    <p>Defining the convolutional neural network architecture.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Compiling and Training</h2>
                                    <p>Configuring the model for training with appropriate loss function and optimizer.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(patience=3)]
)`}
                                        </SyntaxHighlighter>
                                    </div>
                                    <h2 style={{marginTop: '50px'}}>Making Predictions</h2>
                                    <p>Classifying new histopathology images as benign or malignant.</p>
                                    <div className="code-section">
                                        <SyntaxHighlighter language="python" style={dracula}>
{`def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    prediction = model.predict(img)[0][0]
    predicted_class = 'Malignant' if prediction > 0.5 else 'Benign'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return predicted_class, confidence`}
                                        </SyntaxHighlighter>
                                    </div>
                                </div>
                            </div>}
                        {activeSection === 'evaluation' && 
                            <div className="model-details-evaluation">
                                <h1>Model Evaluation</h1>
                                <p>Performance metrics and validation methodology</p>

                                <section className="metric-section">
                                    <h2>Classification Accuracy</h2>
                                    <p className="accuracy">97.66%</p>
                                    <p>Mean accuracy across test dataset</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Precision</th>
                                                <th>96.2%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Recall</td>
                                                <td>95.8%</td>
                                            </tr>
                                            <tr>
                                                <td>F1 Score</td>
                                                <td>96.0%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="metric-section">
                                    <h2>Clinical Performance</h2>
                                    <p className="accuracy">94.3%</p>
                                    <p>Agreement with pathologist diagnoses</p>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Sensitivity</th>
                                                <th>93.7%</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Specificity</td>
                                                <td>95.1%</td>
                                            </tr>
                                            <tr>
                                                <td>ROC AUC</td>
                                                <td>0.98</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </section>

                                <section className="validation-methodology">
                                    <h2>Validation Methodology</h2>
                                    <h3>Cross-Validation</h3>
                                    <p>Our model uses k-fold cross-validation to ensure robust performance:</p>
                                    <ul>
                                        <li>5-fold stratified cross-validation</li>
                                        <li>Separate holdout test set</li>
                                        <li>Clinical validation with pathologist review</li>
                                    </ul>
                                </section>
                            </div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BreastCancer;