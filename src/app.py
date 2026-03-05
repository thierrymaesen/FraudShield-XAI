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
            X_new = df_new.drop('Class', axis=1) if 'Class' in df_new.columns else df_new.copy()
            X_scaled = X_new.copy()
            if 'Time' in X_scaled.columns and 'Amount' in X_scaled.columns:
                X_scaled[['Time', 'Amount']] = scaler.transform(X_scaled[['Time', 'Amount']])
            
            probabilites = xgb_model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilites > 0.99).astype(int)
            df_new['Alerte_Fraude'] = predictions
            nb_fraudes = sum(predictions == 1)
            
            st.divider()
            st.subheader("🚨 Résultats de l'analyse")
            
            if nb_fraudes > 0:
                st.warning(f"⚠️ {nb_fraudes} fraudes détectées !")
                st.dataframe(df_new[df_new['Alerte_Fraude'] == 1])
                
                st.subheader("🧠 Explainable AI : Pourquoi cette alerte ?")
                st.markdown("Analyse détaillée de la première fraude détectée :")
                
                idx_fraude = df_new[df_new['Alerte_Fraude'] == 1].index[0]
                ligne_fraude = X_scaled.iloc[[idx_fraude]]
                
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(ligne_fraude)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                     base_values=explainer.expected_value, 
                                                     data=ligne_fraude.iloc[0], 
                                                     feature_names=ligne_fraude.columns), show=False)
                
                st.pyplot(fig)
                
                st.info("""
**💡 Comment lire ce graphique (Waterfall SHAP) ?**
- **$E[f(x)]$ (en bas)** : Le risque de base moyen.
- **$f(x)$ (en haut)** : Le score de risque final de cette transaction.
- 🔴 **Barres Rouges** : Les variables qui ont poussé l'IA à déclencher l'alerte fraude.
- 🔵 **Barres Bleues** : Les variables qui rassuraient l'IA.

*En entreprise, l'Explainable AI est une obligation légale (RGPD) qui permet aux investigateurs de comprendre la décision de la machine.*
""")
                
            else:
                st.success("✅ Aucune fraude détectée sur ces transactions.")
else:
    st.info("👈 Uploadez un fichier CSV pour démarrer.")
    # --- FOOTER ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 **Créé par Thierry Maesen - 2026**")
    st.sidebar.markdown("[🔗 Mon Profil GitHub](https://github.com/thierrymaesen)")
    st.sidebar.markdown("[🔗 Mon Profil LinkedIn](https://www.linkedin.com/in/thierrymaesen)") 









