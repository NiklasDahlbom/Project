import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("kc_house_data.csv")
    df = df.dropna()
    df = df[df['price'] > 0]
    return df

df = load_data()

# Features used
features = ["bedrooms", "bathrooms", "floors", "sqft_living", "grade", "zipcode"]

# Prepare data
X = df[features]
y = df["price"]

# Amplify feature importance
X["bathrooms"] *= 2
X["floors"] *= 1.5
X["bedrooms"] *= 1.5

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar: Dataset info
st.sidebar.header("📊 Dataset Summary")
st.sidebar.markdown(f"**Total Listings:** {len(df)}")
st.sidebar.markdown(f"**Average Price:** ${df['price'].mean():,.0f}")
st.sidebar.markdown(f"**Min Price:** ${df['price'].min():,.0f}")
st.sidebar.markdown(f"**Max Price:** ${df['price'].max():,.0f}")

# Main Title
st.title("🏠 House Price Predictor")
st.markdown("Enter the features of a house to estimate its selling price.")

# User Inputs
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3,
                           help="Total number of bedrooms in the house")

bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=5.0, value=2.0, step=0.25,
                            help="Total number of bathrooms (e.g., 1.5 = one full + one half)")

floors = st.number_input("Number of Floors", min_value=1, max_value=3, value=1,
                         help="Number of levels in the home (excluding basement)")

sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1500,
                              help="Total finished living space in square feet")

grade = st.slider("Grade (1–13)", min_value=1, max_value=13, value=7,
                  help="Overall construction and design quality (higher is better)")

zipcode = st.selectbox("Zipcode", options=sorted(df["zipcode"].unique()),
                       help="Location of the house (ZIP code)")

# Prediction
input_df = pd.DataFrame([[bedrooms, bathrooms, floors, sqft_living, grade, zipcode]],
                        columns=features)

predicted_price = model.predict(input_df)[0]
avg_price_zip = df[df["zipcode"] == zipcode]["price"].mean()

st.markdown(f"### 💰 Estimated Price: **${predicted_price:,.0f}**")
st.info(f"📍 Average price in {zipcode}: ${avg_price_zip:,.0f}")

# Comparison plot
st.subheader("📉 Predicted vs. Average Price in Zipcode")
fig2, ax2 = plt.subplots()
bar_labels = ['Predicted', 'Average']
bar_values = [predicted_price, avg_price_zip]
ax2.bar(bar_labels, bar_values, color=['green', 'blue'])
ax2.set_ylabel("Price ($)")
ax2.set_title("Price Comparison")
st.pyplot(fig2)

# Feature importance
st.subheader("📊 Feature Importance (Model Learned)")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.write(importance_df.style.format({"Importance": "{:.2%}"}))
