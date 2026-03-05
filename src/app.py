import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

st.title("🛡️ FraudShield AI - Système de Détection")
st.markdown("Ce tableau de bord utilise XGBoost pour détecter les fraudes et **SHAP (Explainable AI)** pour justifier ses décisions.")

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
    st.write("Aperçu des données :", df_new.head(3))
    
    if st.button("Lancer l'analyse 🚀"):
        with st.spinner('Analyse et génération des explications en cours...'):
            # Préparation des données
            X_new = df_new.drop('Class', axis=1) if 'Class' in df_new.columns else df_new.copy()
            X_scaled = X_new.copy()
            if 'Time' in X_scaled.columns and 'Amount' in X_scaled.columns:
                X_scaled[['Time', 'Amount']] = scaler.transform(X_scaled[['Time', 'Amount']])
            
            # Prédictions (Seuil strict à 99% pour éviter les faux positifs)
            probabilites = xgb_model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilites > 0.99).astype(int)
            df_new['Alerte_Fraude'] = predictions
            nb_fraudes = sum(predictions == 1)
            
            st.divider()
            st.subheader("🚨 Résultats de l'analyse")
            
            if nb_fraudes > 0:
                st.warning(f"⚠️ {nb_fraudes} fraudes détectées !")
                st.dataframe(df_new[df_new['Alerte_Fraude'] == 1])
                
                # --- EXPLAINABLE AI (SHAP) ---
                st.subheader("🧠 Explainable AI : Pourquoi cette alerte ?")
                st.markdown("Analyse détaillée de la première fraude détectée (pour les investigateurs) :")
                
                # Récupérer l'index de la première fraude
                idx_fraude = df_new[df_new['Alerte_Fraude'] == 1].index[0]
                ligne_fraude = X_scaled.iloc[[idx_fraude]]
                
                # Calculer les valeurs SHAP
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(ligne_fraude)
                
                # Créer le graphique
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                     base_values=explainer.expected_value, 
                                                     data=ligne_fraude.iloc[0], 
                                                     feature_names=ligne_fraude.columns), show=False)
                
                st.pyplot(fig)
                st.info("💡 **Comment lire ce graphique ?** Les barres rouges montrent les variables qui ont poussé le modèle à crier à la fraude (ex: montant trop élevé, variable V atypique). Les barres bleues montrent ce qui le rassurait.")
                
            else:
                st.success("✅ Aucune fraude détectée sur ces transactions.")
else:
    st.info("👈 Uploadez le fichier sample_transactions.csv pour démarrer.")
