"""
FraudShield XAI - Détection de fraude par Intelligence Artificielle Explicable.
Ce script est le point d'entrée de l'application web Streamlit.
Il charge un modèle XGBoost pré-entraîné, analyse les données de transaction bancaire,
et utilise la librairie SHAP (Explainable AI) pour justifier visuellement chaque alerte de fraude.
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION DE L'APPLICATION
# ==========================================
# Définit le titre de l'onglet du navigateur et la mise en page (large pour mieux afficher les graphiques)

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

st.title("🛡️ FraudShield AI - Système de Détection")
st.markdown("Ce tableau de bord utilise XGBoost pour détecter les fraudes et **SHAP (Explainable AI)** pour justifier ses décisions.")

# ==========================================
# 2. CHARGEMENT DU MODÈLE ET MISE EN CACHE
# ==========================================
# L'utilisation de @st.cache_resource empêche de recharger les lourds fichiers .pkl 
# à chaque interaction de l'utilisateur sur la page web.

@st.cache_resource
def load_models(): 
    # Chargement du modèle XGBoost et du Scaler (pour normaliser les montants/temps)
    return joblib.load('models/xgboost_fraud_model.pkl'), joblib.load('models/scaler.pkl') 

try:
    xgb_model, scaler = load_models()
    st.sidebar.success("✅ Modèle chargé !")
except Exception as e:
    st.error("Erreur de chargement du modèle.")
    st.stop()  # Bloque l'application si les fichiers n'existent pas

# ==========================================
# 3. INTERFACE UTILISATEUR & UPLOAD
# ==========================================

st.sidebar.header("📥 Upload")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture des données importées
    df_new = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", df_new.head(3))
    
    if st.button("Lancer l'analyse 🚀"):
        with st.spinner('Analyse et génération des explications en cours...'):

            # ==========================================
            # 4. PRÉTRAITEMENT DES DONNÉES
            # ==========================================
            # On retire la colonne 'Class' (la réponse) si elle est présente pour éviter de biaiser l'IA
            
            X_new = df_new.drop('Class', axis=1) if 'Class' in df_new.columns else df_new.copy()
            X_scaled = X_new.copy()

            # Application de la standardisation sur les variables "Time" et "Amount"
            # C'est indispensable pour que le modèle puisse interpréter ces valeurs correctement
            
            if 'Time' in X_scaled.columns and 'Amount' in X_scaled.columns:
                X_scaled[['Time', 'Amount']] = scaler.transform(X_scaled[['Time', 'Amount']])

            # ==========================================
            # 5. PRÉDICTIONS DE FRAUDE
            # ==========================================
            # On extrait la probabilité de la classe 1 (Fraude)
            
            probabilites = xgb_model.predict_proba(X_scaled)[:, 1]

            # Seuil de décision ultra-strict : l'IA ne déclenche une alerte que si elle est sûre à > 99%
            # (Ajustement très courant en finance pour éviter les faux positifs)
            
            predictions = (probabilites > 0.99).astype(int)
            df_new['Alerte_Fraude'] = predictions
            nb_fraudes = sum(predictions == 1)
            
            st.divider()
            st.subheader("🚨 Résultats de l'analyse")
            
            if nb_fraudes > 0:
                st.warning(f"⚠️ {nb_fraudes} fraudes détectées !")

                # Affichage des transactions identifiées comme frauduleuses
                
                st.dataframe(df_new[df_new['Alerte_Fraude'] == 1])

                # ==========================================
                # 6. EXPLAINABLE AI (SHAP)
                # ==========================================
                
                st.subheader("🧠 Explainable AI : Pourquoi cette alerte ?")
                st.markdown("Analyse détaillée de la première fraude détectée :")

                # On isole la première transaction marquée comme fraude
                
                idx_fraude = df_new[df_new['Alerte_Fraude'] == 1].index[0]
                ligne_fraude = X_scaled.iloc[[idx_fraude]]

                # Initialisation de l'explicateur SHAP spécifique aux arbres (XGBoost)
                
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(ligne_fraude)

                # Création du graphique en cascade (Waterfall plot)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                     base_values=explainer.expected_value, 
                                                     data=ligne_fraude.iloc[0], 
                                                     feature_names=ligne_fraude.columns), show=False)
                
                st.pyplot(fig)

                # Explications métier pour l'utilisateur final
                
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
# ==========================================
# 7. PIED DE PAGE (FOOTER)
# ==========================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 **Créé par Thierry Maesen - 2026**")
    st.sidebar.markdown("[🔗 Mon Profil GitHub](https://github.com/thierrymaesen)")
    st.sidebar.markdown("[🔗 Mon Profil LinkedIn](https://www.linkedin.com/in/thierrymaesen)") 










