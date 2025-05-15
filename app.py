import streamlit as st
import joblib
import pandas as pd


try:
    model = joblib.load("early_diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model ve ölçekleyici yüklendi.")

except FileNotFoundError:
    st.error("Model veya ölçekleyici dosyaları bulunamadı. Lütfen model.py'yi çalıştırarak gerekli dosyaları oluşturun.")
    st.stop()

# Ana başlık
st.title("Erken Diyabet Tahmin Modeli")

st.sidebar.header("Hasta Verileri")
pregnancies = st.sidebar.number_input("Gebelik Sayısı", min_value=0, step=1)
glucose = st.sidebar.number_input("Glikoz Seviyesi", min_value=0, step=1)
blood_pressure = st.sidebar.number_input("Kan Basıncı", min_value=0, step=1)
skin_thickness = st.sidebar.number_input("Cilt Kalınlığı", min_value=0, step=1)
insulin = st.sidebar.number_input("İnsülin Seviyesi", min_value=0, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diyabet Soy Geçmişi Fonksiyonu", min_value=0.0, step=0.01)
age = st.sidebar.number_input("Yaş", min_value=0, step=1)

# Tahmin butonu
if st.sidebar.button("Tahmin Et"):

    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, diabetes_pedigree, age]],
                             columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]  # Diyabet olasılığı

    st.subheader("Tahmin Sonucu")
    if prediction[0] == 1:
        st.error("Hastada diyabet olma olasılığı yüksek.")
    else:
        st.success("Hastada diyabet olma olasılığı düşük.")

    st.progress(probability)
    st.write(f"Diyabet Olasılığı: {probability:.2%}")