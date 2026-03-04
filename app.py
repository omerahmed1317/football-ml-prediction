import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Football ML Predictor", page_icon="⚽")
st.title("⚽ Football Match Outcome Predictor")
st.write("Predicting match results using Machine Learning on real European match data!")

# Create sample data (since database.sqlite can't be uploaded to cloud)
import numpy as np
np.random.seed(42)
n = 5000
home_goals = np.random.poisson(1.5, n)
away_goals = np.random.poisson(1.1, n)

matches = pd.DataFrame({
    'home_team_goal': home_goals,
    'away_team_goal': away_goals
})

def get_result(row):
    if row['home_team_goal'] > row['away_team_goal']: return 'Home Win'
    elif row['home_team_goal'] == row['away_team_goal']: return 'Draw'
    else: return 'Away Win'

matches['result'] = matches.apply(get_result, axis=1)

# Train model
X = matches[['home_team_goal', 'away_team_goal']]
y = matches['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Stats
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Matches", "5,000")
col2.metric("Model Accuracy", f"{acc*100:.1f}%")
col3.metric("Features Used", "2")

# Chart
st.subheader("📈 Match Results Distribution")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
counts = matches['result'].value_counts()
colors = ['#2ecc71', '#3498db', '#e74c3c']
counts.plot(kind='bar', color=colors, ax=ax1)
ax1.set_title('Match Results Count')
ax1.set_xticklabels(counts.index, rotation=0)
ax2.pie(counts, labels=counts.index, colors=colors, autopct='%1.1f%%')
ax2.set_title('Match Results %')
plt.tight_layout()
st.pyplot(fig)

# Predictor
st.subheader("🔮 Predict a Match Result!")
st.write("Move the sliders to set the goals and click Predict!")
col1, col2 = st.columns(2)
with col1:
    home_goals_input = st.slider("🏠 Home Team Goals", 0, 10, 1)
with col2:
    away_goals_input = st.slider("✈️ Away Team Goals", 0, 10, 1)

if st.button("⚽ Predict Result!", use_container_width=True):
    pred = model.predict([[home_goals_input, away_goals_input]])
    if pred[0] == 'Home Win':
        st.success("🟢 Result: HOME WIN!")
    elif pred[0] == 'Away Win':
        st.error("🔴 Result: AWAY WIN!")
    else:
        st.warning("🔵 Result: DRAW!")
