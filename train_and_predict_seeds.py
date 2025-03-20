import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# File paths for datasets
import os
base_path = r"D:\jtcha1\Desktop\Python\csvfiles"
file_paths = {year: os.path.join(base_path, f"cbb{year[-2:]}.csv") for year in [
    "cbb13", "cbb14", "cbb15", "cbb16", "cbb17", "cbb18", "cbb19",
    "cbb20", "cbb21", "cbb22", "cbb23", "cbb24"
]}

# Load and merge datasets
all_data = []
for year, path in file_paths.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    df["YEAR"] = int(year[-2:]) + 2000  # Extract year from filename
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Drop irrelevant columns
data.drop(columns=['RK'], inplace=True, errors='ignore')

# Encode categorical variables
le_conf = LabelEncoder()
data['CONF'] = le_conf.fit_transform(data['CONF'])

# Fill missing values with median (if any)
data.fillna(data.median(numeric_only=True), inplace=True)

# Select features and target
features = ['CONF', 'G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD',
            'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O', '3P_D', 'ADJ_T', 'WAB']
target = 'SEED'

# Scale features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(k_neighbors=2, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Shift SEED values to start from 0
y_train = y_train - 1
y_test = y_test - 1

# Convert target variable to categorical
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the Neural Network model
# Note: Feel free to adjust the Dense parameter e.g. 128 -> 256 -> 512 etc.
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Neural Network Accuracy: {accuracy:.2f}')

# Predict for 2025 season (using 2024 as a proxy)
if 'YEAR' in data.columns:
    data_2025 = data[data['YEAR'] == 2024].copy()
    data_2025[features] = scaler.transform(data_2025[features])
    predictions = model.predict(data_2025[features])
    data_2025['Predicted_Seed'] = np.argmax(predictions, axis=1) + 1  # Convert back to seed labels

    # Select teams that should be contenders (SEED 1-8)
    top_teams = data_2025[data_2025['Predicted_Seed'] <= 8].copy()

    # If still no teams in SEED 1-8, pick best 10 teams based on BARTHAG instead
    if top_teams.empty:
        print("⚠ No high-seed teams predicted. Selecting teams based on efficiency metrics instead.")
        top_teams = data_2025.sort_values(by=['BARTHAG', 'ADJOE', 'ADJDE'], ascending=[False, False, True]).head(10)

    # Sort and print
    top_teams = top_teams.sort_values(by=['Predicted_Seed', 'BARTHAG', 'ADJOE', 'ADJDE'], ascending=[True, False, False, True])
    print("🔮 Top Contenders for 2025 Championship:")
    print(top_teams[['TEAM', 'Predicted_Seed', 'BARTHAG', 'ADJOE', 'ADJDE']].head(10))
    # Save Top Contenders to a CSV for future analysis
if not top_teams.empty:
    top_teams[['TEAM', 'Predicted_Seed', 'BARTHAG', 'ADJOE', 'ADJDE']].head(10).to_csv("top_contenders_2025.csv", index=False)
    print("✅ Saved top 10 contenders to 'top_contenders_2025.csv'")
else:
    print("⚠ No top contenders found. Try adjusting the model.")
