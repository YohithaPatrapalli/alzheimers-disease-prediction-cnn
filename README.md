# Alzheimer’s Disease Prediction Using CNN Models

## Project Overview

This project is a Django-based web application that uses Convolutional Neural Network (CNN) models to predict Alzheimer's disease stages from brain MRI images.

The application provides an interface for uploading MRI images and viewing prediction results.

**Disclaimer:** This project is intended for academic and research purposes only. It is not a substitute for professional medical diagnosis.

## Features

- Brain MRI image upload
- CNN-based image classification
- Alzheimer's disease stage prediction
- Prediction history
- User authentication
- Admin dashboard
- Dataset management
- Model training and evaluation
- Prediction analytics

## Technologies Used

- Python
- Django
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- HTML
- CSS
- JavaScript
- SQLite
- Bootstrap

## Project Structure

```text
Alzheimer's Disease Prediction Using CNN Models/
│
├── alzheimers_prediction/
├── core/
├── dataset_manager/
├── ml_models/
├── prediction/
├── static/
├── templates/
├── Output Screens/
├── manage.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YohithaPatrapalli/alzheimers-disease-prediction-cnn.git
cd alzheimers-disease-prediction-cnn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Apply database migrations

```bash
python manage.py migrate
```

### 6. Create an admin account

```bash
python manage.py createsuperuser
```

### 7. Run the development server

```bash
python manage.py runserver
```

Open the application in your browser:

```text
http://127.0.0.1:8000/
```

## How to Use

1. Start the Django development server.
2. Open the application in your browser.
3. Log in to the application.
4. Navigate to the MRI prediction page.
5. Upload a brain MRI image.
6. Click the prediction button.
7. View the prediction result.

## Output Screenshots

Screenshots of the application are available in the `Output Screens` folder.

## Future Enhancements

- Improve CNN model accuracy through proper evaluation.
- Support additional deep learning architectures.
- Improve prediction visualization.
- Deploy the application to a cloud platform.
- Add comprehensive model performance reports.

## Disclaimer

This application is developed as an academic project. Predictions are experimental and must not be used for clinical decisions or medical treatment.

## Author

**Yohitha Patrapalli**

GitHub: [YohithaPatrapalli](https://github.com/YohithaPatrapalli)
