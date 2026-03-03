import streamlit as st
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football ML Predictor", page_icon="⚽")
st.title("⚽ Football Match Outcome Predictor")
st.write("Predicting match results using Machine Learning on 25,979 real matches!")

conn = sqlite3.connect('database.sqlite')
matches = pd.read_sql("SELECT home_team_goal, away_team_goal FROM Match WHERE home_team_goal IS NOT NULL", conn)

def get_result(row):
    if row['home_team_goal'] > row['away_team_goal']: return 'Home Win'
    elif row['home_team_goal'] == row['away_team_goal']: return 'Draw'
    else: return 'Away Win'

matches['result'] = matches.apply(get_result, axis=1)

X = matches[['home_team_goal', 'away_team_goal']]
y = matches['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("📊 Match Results in Dataset")
fig, ax = plt.subplots()
matches['result'].value_counts().plot(kind='bar', color=['#2ecc71','#3498db','#e74c3c'], ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

st.subheader("🔮 Predict a Match!")
home_goals = st.slider("Home Team Goals", 0, 10, 1)
away_goals = st.slider("Away Team Goals", 0, 10, 1)

if st.button("Predict Result!"):
    pred = model.predict([[home_goals, away_goals]])
    if pred[0] == 'Home Win':
        st.success("🟢 Result: HOME WIN!")
    elif pred[0] == 'Away Win':
        st.error("🔴 Result: AWAY WIN!")
    else:
        st.warning("🔵 Result: DRAW!")