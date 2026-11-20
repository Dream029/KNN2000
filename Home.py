

data = load_data()
model, accuracy = train_model()

# Header
st.title("ระบบทำนายโรคหัวใจด้วย Machine Learning")
st.caption("Heart Disease Prediction using K-Nearest Neighbors (KNN)")

# Accuracy card
st.subheader("ประสิทธิภาพโมเดล KNN")
col_accuracy, col_info = st.columns([1, 3])
with col_accuracy:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
with col_info:
    st.info("โมเดลใช้ข้อมูลชุด Heart3.csv ในการฝึกและทดสอบ โดยแบ่งข้อมูลสำหรับทดสอบ 20%")

st.markdown("""\
<style>
div.stButton > button[kind="primary"] { width: 100%; border-radius: 10px; height: 3em; }
</style>
""", unsafe_allow_html=True)


st.markdown("### ทำนายข้อมูลโรคหัวใจ")
with st.form("heart_prediction_form"):
    st.markdown("#### กรอกข้อมูล")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("อายุ (Age)", min_value=1, max_value=100, value=40)
        sex_label = st.selectbox("เพศ (Sex)", ["หญิง", "ชาย"])
        chest_pain = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType)", [0, 1, 2, 3])
        resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=300, value=120)
    with c2:
        cholesterol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=1000, value=200)
        fasting_bs_label = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร (FastingBS)", ["ต่ำกว่า 120 (0)", "มากกว่า 120 (1)"])
        resting_ecg = st.selectbox("ผลคลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)", [0, 1, 2])
        max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=0, max_value=300, value=150)
    with c3:
        exercise_angina_label = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)", ["ไม่ (0)", "ใช่ (1)"])
        oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=20.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST_Slope", [0, 1, 2, 3])

    submitted = st.form_submit_button("ทำนายผล", type="primary")

if submitted:
    # Values are encoded as numeric categories to match Heart3.csv
    sex = 0 if sex_label == "หญิง" else 1
    fasting_bs = 0 if fasting_bs_label.startswith("ต่ำกว่า") else 1
    exercise_angina = 0 if exercise_angina_label.startswith("ไม่") else 1

    input_data = pd.DataFrame([[
        age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    ]], columns=[
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
        "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
    ])
    prediction = int(model.predict(input_data)[0])
    st.markdown("---")
    st.subheader("ผลการประเมินจากโมเดล")
    if prediction == 1:
        st.warning("โมเดลพบรูปแบบข้อมูลที่สัมพันธ์กับกลุ่มโรคหัวใจ")
    else:
        st.success("โมเดลพบรูปแบบข้อมูลที่สัมพันธ์กับกลุ่มไม่มีโรคหัวใจ")
    st.caption("ผลนี้เป็นการสาธิตการทำงานของ Machine Learning จากชุดข้อมูล ไม่ใช่การวินิจฉัยทางการแพทย์ หากมีอาการหรือกังวลเกี่ยวกับสุขภาพควรปรึกษาบุคลากรทางการแพทย์")

# Images / educational section
st.markdown("---")
st.header("ความรู้เกี่ยวกับหัวใจ")
img1, img2, img3 = st.columns(3)
with img1:
    st.subheader("โครงสร้างหัวใจ")
    st.image("./img/1.jpg", use_container_width=True)
with img2:
    st.subheader("คลื่นไฟฟ้าหัวใจ (ECG)")
    st.image("./img/2.jpg", use_container_width=True)
with img3:
    st.subheader("สุขภาพหัวใจ")
    st.image("./img/3.jpg", use_container_width=True)

# Dataset table
st.markdown("---")
st.header("สถิติข้อมูลโรคหัวใจ")
translated = data.rename(columns={
    "Age": "Age", "Sex": "Sex", "ChestPainType": "ChestPainType",
    "RestingBP": "RestingBP", "Cholesterol": "Cholesterol",
    "FastingBS": "FastingBS", "RestingECG": "RestingECG",
    "MaxHR": "MaxHR", "ExerciseAngina": "ExerciseAngina",
    "Oldpeak": "Oldpeak", "ST_Slope": "ST_Slope", "HeartDisease": "HeartDisease"
})
st.caption(f"จำนวนข้อมูลทั้งหมด {len(data):,} รายการ")
st.dataframe(translated.head(10), use_container_width=True, hide_index=True)

if st.button("แสดงข้อมูลสรุป"):
    st.write(data.describe())


# Project author
st.markdown("---")
st.header("ผู้จัดทำโครงงาน")
left, right = st.columns([1, 2])
with left:
    st.image("./img/my.jpg", width=220)
with right:
    st.markdown("**ผู้จัดทำ**")
    st.write("โครงงานระบบทำนายโรคหัวใจด้วย Machine Learning (KNN)")
    st.write("พัฒนาโดยใช้ Python, Streamlit และ Scikit-learn")
