import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️")
st.title("🛡️ FraudShield AI - Système de Détection")

@st.cache_resource
def load_models():
    return joblib.load('models/xgboost_fraud_model.pkl'), joblib.load('models/scaler.pkl')

try:
    xgb_model, scaler = load_models()
    st.sidebar.success("✅ Modèle chargé !")
except Exception as e:
    st.error("Erreur de chargement du modèle.")
    st.stop()

st.sidebar.header("📥 Upload")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.write("Aperçu :", df_new.head())
    
    if st.button("Lancer l'analyse 🚀"):
        with st.spinner('Analyse en cours...'):
            X_new = df_new.drop('Class', axis=1) if 'Class' in df_new.columns else df_new.copy()
            if 'Time' in X_new.columns and 'Amount' in X_new.columns:
                X_new[['Time', 'Amount']] = scaler.transform(X_new[['Time', 'Amount']])
            
            # predictions = xgb_model.predict(X_new)
            probabilites = xgb_model.predict_proba(X_new)[:, 1] # Récupère le % de certitude de fraude
            predictions = (probabilites > 0.99).astype(int)    # Seuil ultra-strict à 99%

            nb_fraudes = sum(predictions == 1)
            
            st.subheader("🚨 Résultats")
            if nb_fraudes > 0:
                st.warning(f"⚠️ {nb_fraudes} fraudes détectées !")
            else:
                st.success("✅ Aucune fraude détectée.")
else:
    st.info("👈 Uploadez un fichier CSV pour démarrer.")

