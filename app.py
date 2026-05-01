import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# ============================================================
# PAGE CONFIG + MODERN UI
# ============================================================

st.set_page_config(page_title="Framingham ML Dashboard", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
[data-testid="metric-container"] {
    background-color: #1c1f26;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ Framingham Heart Disease ML Dashboard")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("framingham.csv")
    return df

@st.cache_data
def preprocess(df):
    df = df.drop_duplicates()

    cols_missing = ['education', 'cigsPerDay', 'BPMeds',
                    'totChol', 'BMI', 'heartRate', 'glucose']

    imputer = SimpleImputer(strategy='median')
    df[cols_missing] = imputer.fit_transform(df[cols_missing])
    return df


df = load_data()
df = preprocess(df)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Go to:", ["EDA", "Models", "Prediction", "Chatbot"])

# ============================================================
# EDA
# ============================================================

if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots()
        df.hist(ax=ax, bins=20)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# ============================================================
# MODELS (WITH SUB NAVIGATION)
# ============================================================

elif section == "Models":
    st.header("🤖 Models Dashboard")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["KNN", "Decision Tree", "Logistic Regression", "Comparison"]
    )

    X = df[['age','sysBP','totChol','BMI','glucose']]
    y = df['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def evaluate_model(model, name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        st.subheader(f"📌 {name} Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(acc, 3))
        col2.metric("Precision", round(prec, 3))
        col3.metric("Recall", round(rec, 3))
        col4.metric("F1-score", round(f1, 3))

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        return acc

    # ========================================================
    # INDIVIDUAL MODELS
    # ========================================================

    if model_choice == "KNN":
        k = st.slider("Select K", 1, 10, 5)
        model = KNeighborsClassifier(n_neighbors=k)
        evaluate_model(model, f"KNN (k={k})")

    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
        evaluate_model(model, "Decision Tree")

    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        evaluate_model(model, "Logistic Regression")

    elif model_choice == "Comparison":

        st.subheader("📊 Model Comparison")

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results[name] = accuracy_score(y_test, pred)

        fig, ax = plt.subplots()
        ax.bar(results.keys(), results.values())
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# ============================================================
# PREDICTION
# ============================================================

elif section == "Prediction":

    st.header("🧠 Heart Health Coach + Risk Prediction")

    # ========================================================
    # USER INPUTS
    # ========================================================

    age = st.slider("Age", 30, 80, 50)
    sysBP = st.slider("Systolic BP", 90, 200, 120)
    totChol = st.slider("Cholesterol", 100, 400, 200)
    BMI = st.slider("BMI", 15.0, 40.0, 25.0)
    glucose = st.slider("Glucose", 50, 200, 80)

    X = df[['age','sysBP','totChol','BMI','glucose']]
    y = df['TenYearCHD']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)

    if st.button("Predict Risk"):

        patient = np.array([[age, sysBP, totChol, BMI, glucose]])
        patient_scaled = scaler.transform(patient)

        pred = model.predict(patient_scaled)[0]
        proba = model.predict_proba(patient_scaled)[0][1]

        risk_percent = proba * 100

        st.subheader("📊 Prediction Result")
        st.write(f"🧮 Risk Probability: **{risk_percent:.1f}%**")

        st.subheader("📌 Risk Level & Explanation")

        advice = []

        if risk_percent >= 60:
            st.error("🔴 High Risk")
            advice = [
                "🚨 Immediate lifestyle changes required",
                "🏥 Consult a doctor as soon as possible",
                "🥗 Strict low-salt, low-fat diet",
                "🚶 Daily moderate exercise (30–45 min)",
                "📉 Weight reduction recommended",
                "🩺 Monitor BP and glucose regularly",
                "🚭 Avoid smoking and alcohol",
                "😴 Improve sleep (7–8h)"
            ]

        elif risk_percent >= 30:
            st.warning("🟠 Moderate Risk")
            advice = [
                "⚠️ Preventive actions recommended",
                "🥗 Improve diet (less sugar & fat)",
                "🚶 Regular physical activity",
                "📊 Monitor blood pressure",
                "🍎 Increase fruits & vegetables",
                "🧘 Reduce stress",
                "🏥 Routine check-up advised"
            ]

        else:
            st.success("🟢 Low Risk")
            advice = [
                "✅ Maintain healthy lifestyle",
                "🏃 Keep exercising",
                "🥗 Balanced diet",
                "💧 Stay hydrated",
                "😴 Good sleep habits"
            ]

        st.subheader("🧠 Personalized Health Coach Advice")
        for a in advice:
            st.write(a)

        st.subheader("📈 Detailed Interpretation")
        if pred == 1:
            st.write("Higher likelihood of cardiovascular disease detected.")
        else:
            st.write("Low likelihood of cardiovascular disease detected.")

# ========================================================
# 💬 ADVANCED HEART HEALTH CHATBOT
# ========================================================

elif section == "Chatbot":

    st.header("💬 Heart Health Assistant")

    # ⚠️ Disclaimer médical
    st.info("⚠️ This chatbot provides general advice only. Always consult a doctor for medical decisions.")

    # Mémoire
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about symptoms, risk, diet, or heart health...")

    if user_input:

        st.session_state.messages.append({"role": "user", "content": user_input})

        text = user_input.lower()

        # =========================
        # 🧠 SMART RULE ENGINE
        # =========================

        response = ""

        # 🚨 URGENT SYMPTOMS
        if any(word in text for word in ["chest pain", "douleur poitrine", "shortness of breath", "dizziness"]):
            response = """
🚨 **Possible serious symptoms detected**

These may indicate a heart problem.
👉 Please seek medical help immediately or go to the nearest hospital.

Do not ignore these symptoms.
"""

        # ❤️ BLOOD PRESSURE
        elif any(word in text for word in ["bp", "blood pressure", "hypertension"]):
            response = """
🩺 **Blood Pressure Advice**

- Normal: ~120/80 mmHg  
- High BP increases heart risk  

✅ Reduce salt  
✅ Exercise regularly  
✅ Manage stress  
✅ Check BP frequently  
"""

        # 🧬 CHOLESTEROL
        elif "cholesterol" in text:
            response = """
🧬 **Cholesterol Management**

High cholesterol can block arteries.

✅ Eat fiber (vegetables, oats)  
❌ Avoid fried & fatty foods  
✅ Exercise regularly  
"""

        # 🏃 EXERCISE
        elif any(word in text for word in ["exercise", "sport", "activity"]):
            response = """
🏃 **Physical Activity**

👉 At least 30 min/day recommended  

Good options:
- Walking  
- Cardio  
- Light jogging  

Consistency is key!
"""

        # 🥗 DIET
        elif any(word in text for word in ["diet", "food", "eat"]):
            response = """
🥗 **Heart-Healthy Diet**

✅ Fruits & vegetables  
✅ Lean protein  
✅ Whole grains  

❌ Sugar  
❌ Salt  
❌ Fried foods  
"""

        # 📊 RISK EXPLANATION
        elif "risk" in text:
            response = """
📊 **Heart Disease Risk Factors**

- Age  
- Blood pressure  
- Cholesterol  
- Smoking  
- Diabetes  
- BMI  

👉 Your ML model uses these to predict risk.
"""

        # 🧪 GLUCOSE / DIABETES
        elif any(word in text for word in ["diabetes", "glucose", "sugar"]):
            response = """
🧪 **Diabetes & Heart Risk**

High glucose increases cardiovascular risk.

✅ Control sugar levels  
✅ Healthy diet  
✅ Regular checkups  
"""

        # 👋 GREETING
        elif any(word in text for word in ["hi", "hello", "salem", "hey"]):
            response = "👋 Hello! I can help you understand heart health, symptoms, and prevention."

        # ❓ DEFAULT
        else:
            response = """
🤖 I can help you with:
- Symptoms
- Blood pressure
- Cholesterol
- Diet
- Exercise
- Heart disease risk

Try asking: "What causes heart disease?"
"""

        # =========================
        # 💾 SAVE RESPONSE
        # =========================

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)