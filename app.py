# streamlit UI
import streamlit as st
import pandas as pd
import numpy as np
import pickle



#model de-serialization or loading
with open('Linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# model.predict(data)
# import joblib
# file = "model.pkl"
# model = joblib.load(file)
# model.predict(data)

#encoder de-serialization or loading
with open('labelencoder.pkl', 'rb') as file1:
    encoder = pickle.load(file1)

df = pd.read_csv("cleaned_bengaluru.csv")

st.set_page_config(page_title = "House Price Prediction in Bengaluru",
                   page_icon= "house.png")

with st.sidebar:
    st.title("Bengaluru House Price Prediction")
    st.image("house.png")

#input fields
location = st.selectbox("location", options= df['location'].unique())
BHK = st.selectbox("BHK", options= df['BHK'].unique())
sqft = st.number_input("Total sqft", min_value = 300)
bath = st.selectbox("no. of Restrooms", options= df['bath'].unique())

#encoded new location
encoded_loc = encoder.transform([location])

#new data prep
new_data = [[BHK, sqft, bath, encoded_loc[0]]]

#prediction
col1, col2 = st.columns([1,2])
#predict press button
if col2.button("predict price"):
    pred = model.predict(new_data)[0]
    st.subheader(f"Predicted price: Rs. {round(pred*100000)}")
    