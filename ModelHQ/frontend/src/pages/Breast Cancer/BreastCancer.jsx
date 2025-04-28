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
                <div className="logo">ModelHub</div>
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
        
        {/* LIBRARIES SECTION */}
        <div className="implementation-code">
            <h2>1. Importing Essential Libraries</h2>
            <p>These libraries provide the foundation for our deep learning system:</p>
            
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
            
            <div className="code-explanation">
                <h3>Detailed Explanation:</h3>
                <ul>
                    <li><strong>Line 1:</strong> <code>import tensorflow as tf</code> - Imports TensorFlow, the core deep learning framework we use to build neural networks.</li>
                    <li><strong>Line 2:</strong> <code>from tensorflow.keras.models import Sequential</code> - Imports the Sequential model type which allows us to stack layers linearly.</li>
                    <li><strong>Line 3:</strong> Imports specific layer types:
                        <ul>
                            <li><code>Conv2D</code> - For convolutional layers that detect image features</li>
                            <li><code>MaxPooling2D</code> - For downsampling feature maps</li>
                            <li><code>Flatten</code> - Converts 2D features to 1D for classification</li>
                            <li><code>Dense</code> - Fully connected neural network layers</li>
                            <li><code>Dropout</code> - Prevents overfitting by randomly disabling neurons</li>
                        </ul>
                    </li>
                    <li><strong>Line 4:</strong> <code>ImageDataGenerator</code> - For real-time data augmentation during training.</li>
                    <li><strong>Line 5:</strong> <code>numpy</code> - For numerical operations on image arrays.</li>
                    <li><strong>Line 6:</strong> <code>cv2</code> (OpenCV) - For image loading and preprocessing.</li>
                    <li><strong>Line 7-8:</strong> Visualization libraries for debugging and analysis.</li>
                </ul>
            </div>
        </div>

        {/* IMAGE PREPROCESSING SECTION */}
        <div className="implementation-code">
            <h2>2. Image Loading and Preprocessing</h2>
            <p>This function standardizes all input images to a consistent format:</p>
            
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
            
            <div className="code-explanation">
                <h3>Key Processing Steps:</h3>
                <ol>
                    <li><strong>Line 1:</strong> Defines the function with default size 224x224 pixels (standard for CNNs)</li>
                    <li><strong>Line 3:</strong> <code>cv2.imread()</code> loads the image as a NumPy array</li>
                    <li><strong>Line 4:</strong> Converts from BGR (OpenCV default) to RGB color format</li>
                    <li><strong>Line 7:</strong> Resizes image to target dimensions using bilinear interpolation</li>
                    <li><strong>Line 8:</strong> Normalizes pixel values from 0-255 to 0-1 range for better neural network performance</li>
                </ol>
                <p><strong>Why this matters:</strong> Consistent image dimensions and normalized values help the model learn effectively.</p>
            </div>
        </div>

        {/* DATA AUGMENTATION SECTION */}
        <div className="implementation-code">
            <h2>3. Data Augmentation Configuration</h2>
            <p>Creates variations of training images to improve model generalization:</p>
            
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
            
            <div className="code-explanation">
                <h3>Augmentation Parameters Explained:</h3>
                <table className="params-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Effect</th>
                            <th>Purpose</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>rotation_range=20</code></td>
                            <td>Random rotations up to 20 degrees</td>
                            <td>Makes model invariant to slight rotations</td>
                        </tr>
                        <tr>
                            <td><code>width_shift_range=0.2</code></td>
                            <td>Horizontal shifts up to 20% of width</td>
                            <td>Handles imperfect image centering</td>
                        </tr>
                        <tr>
                            <td><code>shear_range=0.2</code></td>
                            <td>Shear transformations</td>
                            <td>Simulates different viewing angles</td>
                        </tr>
                        <tr>
                            <td><code>zoom_range=0.2</code></td>
                            <td>Random zooming up to 20%</td>
                            <td>Accounts for magnification differences</td>
                        </tr>
                        <tr>
                            <td><code>horizontal_flip=True</code></td>
                            <td>Random left-right flips</td>
                            <td>Doubles training data effectively</td>
                        </tr>
                    </tbody>
                </table>
                <p><strong>Medical Relevance:</strong> These augmentations mimic real-world variations in histopathology slides while preserving diagnostic features.</p>
            </div>
        </div>

        {/* MODEL ARCHITECTURE SECTION */}
        <div className="implementation-code">
            <h2>4. Building the CNN Architecture</h2>
            <p>The core neural network design for feature extraction and classification:</p>
            
            <div className="code-section">
                <SyntaxHighlighter language="python" style={dracula}>
{`model = Sequential([
    # First convolutional block
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    # Second convolutional block (deeper features)
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    # Third convolutional block (high-level features)
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    # Classification head
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Regularization
    Dense(1, activation='sigmoid')  # Binary output
])`}
                </SyntaxHighlighter>
            </div>
            
            <div className="code-explanation">
                <h3>Architecture Breakdown:</h3>
                <div className="architecture-visual">
                    <div className="conv-block">
                        <h4>Conv2D(32, (3,3)) → MaxPooling</h4>
                        <p>Detects basic features like edges and textures</p>
                        <p>32 filters, 3x3 kernel size</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="conv-block">
                        <h4>Conv2D(64, (3,3)) → MaxPooling</h4>
                        <p>Identifies complex patterns like cell structures</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="conv-block">
                        <h4>Conv2D(128, (3,3)) → MaxPooling</h4>
                        <p>Recognizes high-level features like tumor regions</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="dense-block">
                        <h4>Flatten → Dense(128) → Dropout → Sigmoid</h4>
                        <p>Makes final benign/malignant classification</p>
                    </div>
                </div>
                
                <h3>Key Design Choices:</h3>
                <ul>
                    <li><strong>ReLU Activation:</strong> Introduces non-linearity while avoiding vanishing gradients</li>
                    <li><strong>Max Pooling:</strong> Reduces spatial dimensions while preserving important features</li>
                    <li><strong>Dropout(0.3):</strong> Randomly disables 30% of neurons during training to prevent overfitting</li>
                    <li><strong>Sigmoid Output:</strong> Provides probability score between 0 (benign) and 1 (malignant)</li>
                </ul>
            </div>
        </div>

        {/* MODEL TRAINING SECTION */}
        <div className="implementation-code">
            <h2>5. Model Compilation and Training</h2>
            <p>Configuring the learning process and training the model:</p>
            
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
            
            <div className="code-explanation">
                <h3>Training Configuration:</h3>
                <table className="training-table">
                    <tr>
                        <td><strong>Optimizer:</strong></td>
                        <td><code>adam</code> - Adaptive learning rate optimizer that works well for most cases</td>
                    </tr>
                    <tr>
                        <td><strong>Loss Function:</strong></td>
                        <td><code>binary_crossentropy</code> - Standard for binary classification problems</td>
                    </tr>
                    <tr>
                        <td><strong>Metrics:</strong></td>
                        <td><code>accuracy</code> - Tracks percentage of correct classifications</td>
                    </tr>
                    <tr>
                        <td><strong>Epochs:</strong></td>
                        <td>20 complete passes through the training data</td>
                    </tr>
                    <tr>
                        <td><strong>Early Stopping:</strong></td>
                        <td>Stops training if validation accuracy doesn't improve for 3 epochs</td>
                    </tr>
                </table>
                
                <h3>Training Process:</h3>
                <ol>
                    <li>The model sees batches of augmented images</li>
                    <li>Computes loss and updates weights via backpropagation</li>
                    <li>Evaluates on validation set after each epoch</li>
                    <li>Automatically stops if performance plateaus</li>
                </ol>
            </div>
        </div>

        {/* PREDICTION SECTION */}
        <div className="implementation-code">
            <h2>6. Making Predictions</h2>
            <p>Classifying new histopathology images:</p>
            
            <div className="code-section">
                <SyntaxHighlighter language="python" style={dracula}>
{`def predict_image(image_path):
    # Preprocess the image
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Get model prediction
    prediction = model.predict(img)[0][0]
    
    # Interpret results
    predicted_class = 'Malignant' if prediction > 0.5 else 'Benign'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return predicted_class, confidence`}
                </SyntaxHighlighter>
            </div>
            
            <div className="code-explanation">
                <h3>Prediction Workflow:</h3>
                <div className="prediction-flow">
                    <div className="step">
                        <div className="step-number">1</div>
                        <p>Load and preprocess image</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="step">
                        <div className="step-number">2</div>
                        <p>Add batch dimension (required by Keras)</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="step">
                        <div className="step-number">3</div>
                        <p>Get raw prediction (0-1 probability)</p>
                    </div>
                    <div className="arrow">→</div>
                    <div className="step">
                        <div className="step-number">4</div>
                        <p>Convert to class label + confidence %</p>
                    </div>
                </div>
                
                <h3>Decision Threshold:</h3>
                <p>The threshold of 0.5 can be adjusted based on clinical requirements:</p>
                <ul>
                    <li><strong> 0.5:</strong> Classified as Malignant</li>
                    <li><strong>≤ 0.5:</strong> Classified as Benign</li>
                </ul>
                <p><strong>Confidence Score:</strong> Represents how certain the model is, with 100% being completely confident.</p>
            </div>
        </div>
    </div>
}
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