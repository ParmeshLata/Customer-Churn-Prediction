import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
data=pd.read_csv("Churn_Modelling.csv")
data=data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
label_encoder_gender=LabelEncoder()
data["Gender"]=label_encoder_gender.fit_transform(data["Gender"])
from sklearn.preprocessing import OneHotEncoder
ohe_geography=OneHotEncoder(sparse_output=False)
encoded_data=ohe_geography.fit_transform(data[["Geography"]])
geo_sub_names=ohe_geography.get_feature_names_out(["Geography"])
geo_encoded_df=pd.DataFrame(data=encoded_data, columns=geo_sub_names)
data=pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)
x=data.drop("Exited", axis=1)
y=data["Exited"]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
scaler_x=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
model=Sequential([
    Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
early_stopping_callback=EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history=model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping_callback]
)
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st
st.title("CUSTOMER CHURN PREDICTION")
geography=st.selectbox("Geography", ohe_geography.categories_[0])
gender=st.selectbox("Gender", label_encoder_gender.classes_)
age=st.slider("Age", 18, 92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure", 0, 10)
num_of_products=st.slider("Number of Products")
has_credit_card=st.selectbox("Has Credit Card", [0,1])
is_active_member=st.selectbox("Is Active Member", [0,1])
input_data=pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]})
geography_encoded=ohe_geography.transform([[geography]])
geography_encoded_df=pd.DataFrame(geography_encoded,columns=ohe_geography.get_feature_names_out(["Geography"]))
input_value=pd.concat([input_data.reset_index(drop=True), geography_encoded_df])
scaled_input=scaler_x.transform(input_value)
prediction=model.predict(scaled_input)
prediction_probability=prediction[0][0]
st.write(f"Churn Probability: {prediction_probability:.2f}")
if prediction_probability>0.5:
    st.write("Customer is likely to Churn.")
else:
    st.write("Customer is not likely to churn.")