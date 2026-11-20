import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease KNN", layout="wide")

st.caption("ระบบจำแนกข้อมูลโรคหัวใจโดยใช้ Machine Learning")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("โครงสร้างหัวใจ")
    st.image("./img/1.jpg", use_container_width=True)

with col2:
    st.subheader("คลื่นไฟฟ้าหัวใจ (ECG)")
    st.image("./img/2.jpg", use_container_width=True)

with col3:
    st.subheader("สุขภาพหัวใจ")
    st.image("./img/3.jpg", use_container_width=True)

st.markdown("")

st.subheader("สถิติข้อมูลโรคหัวใจ")
dt = pd.read_csv("./data/Heart3.csv")
st.dataframe(dt.head(10), use_container_width=True)

html_stat = """
<div style="background-color:#EC7063;padding:12px;border-radius:10px;margin-top:10px;margin-bottom:20px;">
<center><h5 style="color:black;margin:0;">สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_stat, unsafe_allow_html=True)

X = dt.drop('HeartDisease', axis=1)
y = dt['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred) * 100

st.subheader("ประสิทธิภาพโมเดล KNN")
st.metric(label="Accuracy", value=f"{acc:.2f}%")

html_predict = """
<div style="background-color:#6BD5DA;padding:12px;border-radius:10px;margin-top:10px;margin-bottom:15px;">
<center><h5 style="color:black;margin:0;">ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_predict, unsafe_allow_html=True)

st.caption("กรุณากรอกข้อมูลเพื่อทำนายผล")

in_col1, in_col2, in_col3 = st.columns(3)

with in_col1:
    age = st.slider("อายุ (Age)", min_value=1, max_value=100, value=22)
    sex = st.selectbox("เพศ (Sex)", options=[0, 1], index=0)
    chest_pain = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType)", options=[1, 2, 3, 4], index=0)
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=300, value=120, step=1)

with in_col2:
    cholesterol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=200, step=1)
    fasting_bs = st.selectbox("FastingBS", options=[0, 1], index=0)
    resting_ecg = st.selectbox("RestingECG", options=[1, 2, 3], index=0)
    max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=40, max_value=250, value=150, step=1)

with in_col3:
    exercise_angina = st.selectbox("ExerciseAngina", options=[0, 1], index=0)
    oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=0.00, step=0.10, format="%.2f")
    st_slope = st.selectbox("ST_Slope", options=[1, 2, 3], index=0)

if st.button("ทำนายผล"):
    x_input = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                         fasting_bs, resting_ecg, max_hr, exercise_angina,
                         oldpeak, st_slope]])
    x_input_scaled = scaler.transform(x_input)
    prediction = knn_model.predict(x_input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (Heart Disease = 1)")
    else:
        st.success("✅ ผลการทำนาย: ปกติ / ไม่มีความเสี่ยงเป็นโรคหัวใจ (Heart Disease = 0)")

st.markdown("---")

st.markdown("<h3 style='text-align: center;'>ผู้จัดทำโครงงาน</h3>", unsafe_allow_html=True)

_, center_col, _ = st.columns([1.2, 1, 1.2])
with center_col:
    with st.container(border=True):
        _, img_col, _ = st.columns([0.5, 1, 0.5])
        with img_col:
            st.image("./img/my.jpg", use_container_width=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <h5 style="margin-bottom: 4px;">ผู้จัดทำโครงงาน</h5>
                <p style="font-size: 13px; color: gray; margin: 0;">สาขาเทคโนโลยีสารสนเทศ</p>
                <p style="font-size: 12px; color: gray; margin-top: 2px;">KNN Heart Disease Prediction</p>
            </div>
            """,
            unsafe_allow_html=True
        )