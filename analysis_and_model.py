import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Предобработка
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        st.subheader("Первые строки датасета")
        st.write(data.head())

        if data.isnull().sum().sum() > 0:
            st.warning("Есть пропущенные значения!")
        else:
            st.success("Пропущенных значений нет.")

        # Масштабирование
        features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Модель
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        st.subheader("Оценка модели")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-кривая")
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

        # Предсказания
        st.subheader("Предсказание отказа")
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        prediction = model.predict(input_df)
        st.write(f"**Оборудование {'откажет' if prediction[0]==1 else 'не откажет'}**")
