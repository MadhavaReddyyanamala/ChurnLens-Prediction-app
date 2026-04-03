import streamlit as st
import pandas as pd
import pickle

# ================== 1. CONFIG ==================
st.set_page_config(
    page_title="ChurnLens Customer Churn Predictor",
    layout="wide",
    page_icon="📊"
)

# ================== 2. UI THEME ==================
PRIMARY_GRADIENT = "linear-gradient(90deg, #4facfe, #00f2fe)"
PAGE_BG = "#F5F9FF"
CARD_BG = "#FFFFFF"

# ================== 3. CSS ==================
st.markdown(f"""
<style>
.stApp {{
    background-color: {PAGE_BG};
}}

.title {{
    background: {PRIMARY_GRADIENT};
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    color: white;
    font-weight: bold;
}}

.card {{
    background: {CARD_BG};
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}}

.good {{
    background: #d4edda;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    color: #155724;
}}

.bad {{
    background: #f8d7da;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    color: #721c24;
}}

footer {{
    text-align: center;
    margin-top: 30px;
    color: gray;
}}
</style>
""", unsafe_allow_html=True)

# ================== 4. LOAD FILES ==================
@st.cache_resource
def load_files():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    cat_cols = pickle.load(open("cat_cols.pkl", "rb"))
    num_cols = pickle.load(open("num_cols.pkl", "rb"))
    feature_order = pickle.load(open("feature_order.pkl", "rb"))

    try:
        target_encoder = pickle.load(open("target_encoder.pkl", "rb"))
    except:
        target_encoder = None

    return model, scaler, label_encoders, cat_cols, num_cols, feature_order, target_encoder


model, scaler, label_encoders, cat_cols, num_cols, feature_order, target_encoder = load_files()

# ================== 5. HEADER ==================
st.markdown("<div class='title'>📊 ChurnLens Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.write("### Enter customer details")

# ================== 6. INPUT ==================
col1, col2 = st.columns(2)
user_inputs = {}

# ---------- CATEGORICAL ----------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Customer Profile")

    for col in cat_cols:
        user_inputs[col] = st.selectbox(
            col.replace("_", " ").title(),
            list(label_encoders[col].classes_)
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- NUMERICAL ----------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Usage & Billing")

    for col in num_cols:
        label = col.replace("_", " ").title()
        key = col.lower().replace("_", "")

        # 🎯 SLIDERS
        if key == "tenure":
            user_inputs[col] = st.slider(label, 0, 72, 12)

        elif key == "monthlycharges":
            user_inputs[col] = st.slider(label, 0.0, 150.0, 50.0)

        elif key == "totalcharges":
            user_inputs[col] = st.slider(label, 0.0, 10000.0, 1000.0)

        # Other numeric
        else:
            user_inputs[col] = st.number_input(label, min_value=0.0, step=1.0)

    st.markdown("</div>", unsafe_allow_html=True)

# ================== 7. DATAFRAME ==================
df = pd.DataFrame([user_inputs])

# ================== 8. PREPROCESS ==================
try:
    for col in cat_cols:
        df[col] = label_encoders[col].transform(df[col])

    df[num_cols] = scaler.transform(df[num_cols])
    df = df.reindex(columns=feature_order)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ================== 9. PREDICT ==================
if st.button("🔍 Predict Churn", use_container_width=True):

    try:
        prediction = model.predict(df)[0]

        # Probability
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("<div class='bad'>⚠️ Customer Likely to Churn</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='good'>✅ Customer Will Stay</div>", unsafe_allow_html=True)

        if prob:
            st.info(f"Churn Probability: **{prob*100:.2f}%**")

        # Decode label (optional)
        if target_encoder:
            try:
                label = target_encoder.inverse_transform([prediction])[0]
                st.write(f"Prediction Label: **{label}**")
            except:
                pass

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ================== 10. FOOTER ==================
st.markdown("<footer>Built with ❤️ using Streamlit</footer>", unsafe_allow_html=True)