# 🛡️ FraudShield AI : Détection de Fraude par Carte Bancaire en Temps Réel

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

> **Application Web Interactive :** [🔗 Tester FraudShield AI ici](https://fraudshield-thierry.streamlit.app/)

## 📖 Vue d'ensemble du Projet

La fraude à la carte bancaire coûte des milliards d'euros chaque année à l'industrie financière. L'enjeu n'est pas seulement financier, mais concerne aussi la confiance des clients. 
**FraudShield AI** est une solution complète de bout en bout (End-to-End) qui utilise le Machine Learning pour analyser les transactions financières et détecter les anomalies avec une haute précision, tout en minimisant les "faux positifs" qui bloquent inutilement les cartes de clients légitimes.

## 🎯 Enjeux Business et Valeur Ajoutée

Dans le domaine bancaire, la détection de fraude fait face à un défi majeur : le déséquilibre extrême des données (environ 0,17% de fraudes pour 99,83% de transactions légitimes).
Ce projet apporte des solutions concrètes à ces problématiques métiers :
- **Réduction des pertes financières :** Identification proactive des transactions frauduleuses avant la compensation.
- **Optimisation de l'expérience client :** Ajustement fin du seuil de probabilité (Threshold à 99%) pour drastiquement réduire les faux positifs (blocages de cartes injustifiés).
- **Aide à la décision (Explainable AI) :** Intégration de graphiques SHAP pour expliquer visuellement aux investigateurs *pourquoi* l'IA a déclenché une alerte (conformité RGPD sur le "Droit à l'explication").

## 🧠 L'Approche Technique & Architecture

L'architecture du projet est divisée en trois piliers principaux :

1. **Traitement des Données (Data Engineering)**
   - Normalisation (Scaling) des variables de temps (`Time`) et de montants (`Amount`) avec `StandardScaler`.
   - Préservation des variables anonymisées issues de l'Analyse en Composantes Principales (PCA).

2. **Modélisation Machine Learning**
   - Utilisation de l'algorithme **XGBoost (Extreme Gradient Boosting)**, réputé pour ses performances sur les données tabulaires fortement déséquilibrées.
   - Utilisation de la probabilité prédictive (`predict_proba`) plutôt qu'une classification binaire stricte pour permettre une flexibilité sur le seuil d'alerte.

3. **Déploiement Cloud (MLOps)**
   - Sérialisation des modèles entraînés via `joblib`.
   - Création d'une interface utilisateur web avec **Streamlit**.
   - Déploiement automatisé et continu (CI/CD) sur **Streamlit Community Cloud** via la synchronisation avec GitHub.

## 📊 Format des Données (Input)

Pour des raisons de confidentialité bancaire, le modèle a été entraîné sur des données européennes dont la majorité des variables ont été transformées par Analyse en Composantes Principales (PCA). 

Pour que l'application fonctionne, le fichier CSV uploadé doit respecter **strictement** cette structure (30 ou 31 colonnes) :

| Nom de la Colonne | Type de donnée | Description |
| :--- | :--- | :--- |
| `Time` | Numérique | Secondes écoulées entre cette transaction et la première du dataset. |
| `V1` à `V28` | Numérique (Float) | Variables anonymisées issues de la transformation PCA. |
| `Amount` | Numérique (Float) | Montant de la transaction. |
| `Class` *(Optionnel)* | Binaire (0 ou 1) | La cible (0 = Normal, 1 = Fraude). L'application l'ignorera. |

> 💡 **Besoin de données pour tester ?** 
> 1. Cliquez sur ce lien : [📄 sample_transactions.csv](https://github.com/thierrymaesen/FraudShield-XAI/blob/main/sample_transactions.csv)
> 2. Sur la page qui s'ouvre, cliquez sur le bouton **Download** (l'icône avec une flèche vers le bas ⬇️) en haut à droite du tableau de données.
> 3. Uploadez ce fichier directement dans l'application pour voir la détection et les graphiques SHAP en action !

## 📂 Structure du Répertoire

```text
📦 FraudShield-XAI
 ┣ 📂 models/                     # Modèles ML sérialisés
 ┃ ┣ 📜 xgboost_fraud_model.pkl   # Le modèle XGBoost entraîné
 ┃ ┗ 📜 scaler.pkl                # Le scaler pour la normalisation
 ┣ 📂 src/                        # Code source de l'application
 ┃ ┗ 📜 app.py                    # Script principal Streamlit
 ┣ 📜 sample_transactions.csv     # Fichier de test (Demo)
 ┣ 📜 requirements.txt            # Dépendances Python
 ┣ 📜 .gitignore                  # Fichiers ignorés par Git
 ┗ 📜 README.md                   # Documentation du projet
🚀 Comment exécuter le projet localement ?
Si vous souhaitez faire tourner cette application sur votre propre machine :

1. Cloner le dépôt :

bash
git clone https://github.com/thierrymaesen/FraudShield-XAI.git
cd FraudShield-XAI
2. Créer un environnement virtuel (recommandé) :

bash
python -m venv venv
source venv/Scripts/activate  # Sur Windows
3. Installer les dépendances :

bash
pip install -r requirements.txt
4. Lancer l'application Streamlit :

bash
streamlit run src/app.py
📈 Améliorations futures (Roadmap)
 Remplacer la technique SMOTE par le paramètre scale_pos_weight natif de XGBoost pour améliorer l'apprentissage sur le dataset original.

 Connecter l'application à une base de données SQL pour historiser les prédictions.

⚖️ Licences et Remerciements
Code Source
Le code de cette application est distribué sous la Licence MIT. Vous êtes libre de l'utiliser, de le modifier et de le distribuer sous réserve d'inclure la notification de copyright originale.

Jeu de Données (Dataset)
Le modèle d'intelligence artificielle a été entraîné en utilisant le jeu de données public "Credit Card Fraud Detection" mis à disposition par le Machine Learning Group (MLG) de l'Université Libre de Bruxelles (ULB).

Source : Kaggle - Credit Card Fraud Detection

Licence des données : Open Data / CC0 Public Domain.

Projet réalisé par Thierry Maesen.
