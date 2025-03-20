# 🏀 NCAA 2025 March Madness Prediction

This project predicts the **2025 NCAA March Madness Champion** using **Machine Learning and Deep Learning**.  
It trains a **Neural Network (MLP) in TensorFlow/Keras** to predict **team seed rankings**, then **simulates tournament matchups** to determine the most likely **NCAA Champion**.

## 🚀 Features

✅ **Machine Learning Model**: Multi-Layer Perceptron (MLP) Neural Network (TensorFlow/Keras)  
✅ **Seed Prediction**: Predicts NCAA team seeds using past data  
✅ **Tournament Simulation**: Runs head-to-head matchups based on **efficiency metrics**  
✅ **Class Balancing**: Uses **SMOTE & Class Weights** to ensure fair training  
✅ **Historical Data**: Trained using **2013-2024 NCAA statistics** from Andrew Sundberg on Kaggle.com https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset/data 

---

## 📊 Model Overview

- **Framework**: TensorFlow / Keras
- **Input Features**: 
  - Adjusted Offensive Efficiency (**ADJOE**)
  - Adjusted Defensive Efficiency (**ADJDE**)
  - Power Rating (**BARTHAG**)
  - Effective Field Goal Percentage (**EFG%**)
  - Free Throw Rate (**FTR**)
  - Rebounding & Turnover Rates
- **Target Output**: `SEED` (1-16) – Predicted ranking for each team

### 🏆 **Final Prediction**: 2025 NCAA Champion

- It will be based on efficiency and tournament simulation.

---

## 🛠 Installation & Usage

### 1️⃣ **Install Dependencies**
Ensure Python 3.x is installed. Then install required libraries:
```sh
pip install -r requirements.txt
```

### 1️⃣ **Run the Neural Network Model**
```sh
python train_and_predict_seeds.py
```
✅ **Saves `top_contenders_2025.csv`** with the best 10 teams.

### 3️⃣ **Simulate the Tournament & Predict the Winner**
```sh
python predict_winner.py
```
🏆 **Prints the 2025 NCAA Champion Prediction!**

---

## 🎯 Future Improvements
- 🏀 I want to try a different algorithm and compare the two. Maybe XGBoost
- 📊 Add player stats and potential injuries for better predictions.

---
