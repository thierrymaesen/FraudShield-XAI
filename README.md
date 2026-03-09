<div align="center">

🇫🇷 [Version française](#french) | 🇬🇧 [English version](#english)

</div>

---

<a name="french"></a>

# 🛡️ FraudShield AI : Détection de Fraude par Carte Bancaire en Temps Réel

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-11557C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

> **Application Web Interactive :** [🔗 Tester FraudShield AI ici](https://fraudshield-thierry.streamlit.app/)

## 📖 Vue d'ensemble du Projet

La fraude à la carte bancaire coûte des milliards d'euros chaque année à l'industrie financière. L'enjeu n'est pas seulement financier, mais concerne aussi la confiance des clients.

**FraudShield AI** est une solution complète de bout en bout (End-to-End) qui utilise le Machine Learning pour analyser les transactions financières et détecter les anomalies avec une haute précision, tout en minimisant les "faux positifs" qui bloquent inutilement les cartes de clients légitimes.

## 🎯 Enjeux Business et Valeur Ajoutée

Dans le domaine bancaire, la détection de fraude fait face à un défi majeur : le déséquilibre extrême des données (environ 0,17% de fraudes pour 99,83% de transactions légitimes). Ce projet apporte des solutions concrètes :

- **Réduction des pertes financières :** Identification proactive des transactions frauduleuses avant la compensation.
- **Optimisation de l'expérience client :** Ajustement fin du seuil de probabilité (Threshold à 99%) pour drastiquement réduire les faux positifs (blocages de cartes injustifiés).
- **Aide à la décision (Explainable AI) :** Intégration de graphiques SHAP pour expliquer visuellement aux investigateurs *pourquoi* l'IA a déclenché une alerte (conformité RGPD sur le "Droit à l'explication").

## 🧠 L'Approche Technique & Architecture

L'architecture du projet est divisée en trois piliers principaux :

1. **Traitement des Données (Data Engineering)**
   - Normalisation (Scaling) des variables de temps et de montants avec `StandardScaler`.
   - Préservation des variables anonymisées issues de l'Analyse en Composantes Principales (PCA).

2. **Modélisation Machine Learning**
   - Utilisation de l'algorithme **XGBoost (Extreme Gradient Boosting)**, réputé pour ses performances sur les données tabulaires fortement déséquilibrées.
   - Utilisation de la probabilité prédictive (`predict_proba`) plutôt qu'une classification binaire stricte.

3. **Déploiement Cloud (MLOps)**
   - Sérialisation des modèles entraînés via `joblib`.
   - Création d'une interface utilisateur web avec **Streamlit**.
   - Déploiement automatisé et continu (CI/CD) sur **Streamlit Community Cloud** via la synchronisation avec GitHub.

## 📊 Format des Données (Input)

Pour des raisons de confidentialité bancaire, le modèle a été entraîné sur des données européennes dont la majorité des variables ont été transformées par Analyse en Composantes Principales (PCA). Le fichier CSV uploadé doit respecter cette structure :

| Nom de la Colonne | Type de donnée | Description |
| :--- | :--- | :--- |
| `Time` | Numérique | Secondes écoulées entre cette transaction et la première du dataset. |
| `V1` à `V28` | Numérique | Variables anonymisées issues de la transformation PCA. |
| `Amount` | Numérique | Montant de la transaction. |
| `Class` | Binaire | (Optionnel) La cible (0 = Normal, 1 = Fraude). |

> 💡 **Besoin de données pour tester ?**
> 1. Cliquez sur ce lien : [📄 sample_transactions.csv](https://github.com/thierrymaesen/FraudShield-XAI/blob/main/sample_transactions.csv)
> 2. Cliquez sur le bouton **Download** (l'icône avec une flèche vers le bas ⬇️) en haut à droite du tableau de données.
> 3. Uploadez ce fichier directement dans l'application !

## 📂 Structure du Répertoire

* 📁 **models/** : Modèles ML sérialisés
  * 📜 `xgboost_fraud_model.pkl` : Le modèle XGBoost entraîné
  * 📜 `scaler.pkl` : Le scaler pour la normalisation
* 📁 **src/** : Code source de l'application
  * 📜 `app.py` : Script principal Streamlit
* 📜 `sample_transactions.csv` : Fichier de test (Demo)
* 📜 `requirements.txt` : Dépendances Python

## 🚀 Comment exécuter le projet localement ?

Si vous souhaitez faire tourner cette application sur votre propre machine :

**1. Cloner le dépôt :**
```bash
git clone https://github.com/thierrymaesen/FraudShield-XAI.git
cd FraudShield-XAI
```

**2. Créer un environnement virtuel (recommandé) :**
```bash
python -m venv venv
source venv/Scripts/activate
```

**3. Installer les dépendances :**
```bash
pip install -r requirements.txt
```

**4. Lancer l'application Streamlit :**
```bash
streamlit run src/app.py
```

## 📈 Améliorations futures (Roadmap)

- [ ] Remplacer la technique SMOTE par le paramètre `scale_pos_weight` natif de XGBoost pour améliorer l'apprentissage sur le dataset original.
- [ ] Connecter l'application à une base de données SQL pour historiser les prédictions.

## ⚖️ Licences et Remerciements

**Code Source**
Le code de cette application est distribué sous la [Licence MIT](https://opensource.org/licenses/MIT).

**Jeu de Données (Dataset)**
Le modèle a été entraîné en utilisant le jeu de données "Credit Card Fraud Detection" mis à disposition par le Machine Learning Group (MLG) de l'Université Libre de Bruxelles (ULB).
- Source : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Licence des données : Open Data / CC0 Public Domain.

Projet réalisé par **Thierry Maesen**.

---

<a name="english"></a>

# 🛡️ FraudShield AI: Real-Time Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-11557C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

> **Interactive Web Application:** [🔗 Try FraudShield AI here](https://fraudshield-thierry.streamlit.app/)

## 📖 Project Overview

Credit card fraud costs billions of euros every year to the financial industry. The challenge is not only financial, but also concerns customer trust.

**FraudShield AI** is a complete End-to-End solution that uses Machine Learning to analyze financial transactions and detect anomalies with high accuracy, while minimizing "false positives" that unnecessarily block legitimate customers' cards.

## 🎯 Business Challenges and Value Proposition

In the banking sector, fraud detection faces a major challenge: extreme data imbalance (approximately 0.17% fraud vs. 99.83% legitimate transactions). This project provides concrete solutions:

- **Reduction of financial losses:** Proactive identification of fraudulent transactions before settlement.
- **Customer experience optimization:** Fine-tuning the probability threshold (Threshold at 99%) to drastically reduce false positives (unjustified card blocks).
- **Decision support (Explainable AI):** Integration of SHAP visualizations to visually explain to investigators *why* the AI triggered an alert (GDPR compliance on the "Right to explanation").

## 🧠 Technical Approach & Architecture

The project architecture is divided into three main pillars:

1. **Data Engineering**
   - Normalization (Scaling) of time and amount variables using `StandardScaler`.
   - Preservation of anonymized variables from Principal Component Analysis (PCA).

2. **Machine Learning Modeling**
   - Use of the **XGBoost (Extreme Gradient Boosting)** algorithm, renowned for its performance on highly imbalanced tabular data.
   - Use of predictive probability (`predict_proba`) rather than strict binary classification.

3. **Cloud Deployment (MLOps)**
   - Serialization of trained models via `joblib`.
   - Web user interface built with **Streamlit**.
   - Automated and continuous deployment (CI/CD) on **Streamlit Community Cloud** via GitHub synchronization.

## 📊 Data Format (Input)

For banking confidentiality reasons, the model was trained on European data where the majority of variables were transformed by Principal Component Analysis (PCA). The uploaded CSV file must follow this structure:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Time` | Numeric | Seconds elapsed between this transaction and the first one in the dataset. |
| `V1` to `V28` | Numeric | Anonymized variables from PCA transformation. |
| `Amount` | Numeric | Transaction amount. |
| `Class` | Binary | (Optional) The target (0 = Normal, 1 = Fraud). |

> 💡 **Need data to test?**
> 1. Click on this link: [📄 sample_transactions.csv](https://github.com/thierrymaesen/FraudShield-XAI/blob/main/sample_transactions.csv)
> 2. Click the **Download** button (the icon with a downward arrow ⬇️) at the top right of the data table.
> 3. Upload this file directly into the application!

## 📂 Repository Structure

* 📁 **models/** : Serialized ML models
  * 📜 `xgboost_fraud_model.pkl` : Trained XGBoost model
  * 📜 `scaler.pkl` : Scaler for normalization
* 📁 **src/** : Application source code
  * 📜 `app.py` : Main Streamlit script
* 📜 `sample_transactions.csv` : Test file (Demo)
* 📜 `requirements.txt` : Python dependencies

## 🚀 How to Run the Project Locally?

If you want to run this application on your own machine:

**1. Clone the repository:**
```bash
git clone https://github.com/thierrymaesen/FraudShield-XAI.git
cd FraudShield-XAI
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit application:**
```bash
streamlit run src/app.py
```

## 📈 Future Improvements (Roadmap)

- [ ] Replace the SMOTE technique with XGBoost's native `scale_pos_weight` parameter to improve learning on the original dataset.
- [ ] Connect the application to a SQL database to store prediction history.

## ⚖️ Licenses and Acknowledgments

**Source Code**
The code of this application is distributed under the [MIT License](https://opensource.org/licenses/MIT).

**Dataset**
The model was trained using the "Credit Card Fraud Detection" dataset provided by the Machine Learning Group (MLG) at the Université Libre de Bruxelles (ULB).
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Data License: Open Data / CC0 Public Domain.

Project by **Thierry Maesen**.
