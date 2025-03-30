# Customer Churn Prediction

This project implements a deep learning model using TensorFlow and Keras to predict customer churn based on various features such as credit score, age, balance, and other factors. The model is trained on the "Churn_Modelling.csv" dataset and deployed using Streamlit.

## Features
- Data preprocessing with label encoding and one-hot encoding.
- Feature scaling using StandardScaler.
- Deep learning model using TensorFlow/Keras.
- Early stopping to prevent overfitting.
- Streamlit-based UI for user input and salary prediction.

## Installation
To run this project, install the required dependencies:
- Scikit-Learn
- TensorFlow
- Streamlit
- Keras

## Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/ParmeshLata/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run churn_app.py
   ```

3. Enter customer details in the UI to get the predicted salary.

## Model Training
The model is trained using the following steps:
- Load and preprocess the dataset.
- Encode categorical variables.
- Scale features using StandardScaler.
- Train a neural network with ReLU activations.
- Monitor training with early stopping.

## Future Enhancements
- Improve model accuracy with hyperparameter tuning.
- Implement a more interactive UI.
- Deploy the model as a web API.

## License
I have developed this project after learning the Deep Learning from https://github.com/krishnaik06.

---
### Author
Developed by  https://github.com/ParmeshLata 🚀
