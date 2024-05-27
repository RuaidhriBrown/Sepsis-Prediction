import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load the preprocessed dataset
agg_data_imputed = pd.read_csv('data/preprocessed_data.csv')

# Feature and target variables
X = agg_data_imputed.drop(columns=['Patient_ID', 'Hour', 'EventualSepsisLabel'])
y = agg_data_imputed['EventualSepsisLabel']

# Feature selection using RandomForest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]  # Select top 10 features

# Print the selected important features
print("Top 10 important features:")
for i in indices:
    print(f"{X.columns[i]}: {importances[i]:.4f}")

X_selected = X.iloc[:, indices]

# Include 'Hour' as a feature
X_selected['Hour'] = agg_data_imputed['Hour']

# Sample the data
sample_fraction = 0.2  # Adjust this fraction as needed
X_sampled, _, y_sampled, _ = train_test_split(X_selected, y, train_size=sample_fraction, stratify=y, random_state=42)

# Balance the dataset
smote = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)

# First, apply SMOTE to oversample the minority class
X_oversampled, y_oversampled = smote.fit_resample(X_sampled, y_sampled)

# Then, apply undersampling to balance the combined dataset
X_resampled, y_resampled = undersample.fit_resample(X_oversampled, y_oversampled)

# Debugging: Check shapes after resampling
print(f"X_resampled shape: {X_resampled.shape}")
print(f"y_resampled shape: {y_resampled.shape}")

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Debugging: Check shapes after train-test split
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN [samples, timesteps, features]
# Assuming 'Hour' as the timestep feature for reshaping
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
rnn_model.add(Dropout(0.5))
rnn_model.add(SimpleRNN(32, return_sequences=False))
rnn_model.add(Dropout(0.5))
rnn_model.add(Dense(1, activation='sigmoid'))

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the RNN model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rnn_history = rnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Save the RNN model
rnn_model.save('rnn_model.keras')

# Evaluate the RNN model
y_pred_prob_rnn = rnn_model.predict(X_test_reshaped).ravel()

# Adjust threshold to reduce false negatives
threshold = 0.3  # Lowered threshold for higher sensitivity
y_pred_rnn = (y_pred_prob_rnn > threshold).astype(int)

# Evaluate metrics for RNN model
accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
precision_rnn = precision_score(y_test, y_pred_rnn)
recall_rnn = recall_score(y_test, y_pred_rnn)
f1_rnn = f1_score(y_test, y_pred_rnn)
roc_auc_rnn = roc_auc_score(y_test, y_pred_prob_rnn)

# Print RNN evaluation metrics
print(f"RNN Model Accuracy: {accuracy_rnn:.2f}")
print(f"RNN Model Precision: {precision_rnn:.2f}")
print(f"RNN Model Recall: {recall_rnn:.2f}")
print(f"RNN Model F1 Score: {f1_rnn:.2f}")
print(f"RNN Model ROC AUC Score: {roc_auc_rnn:.2f}")

# Display RNN classification report
print("\nRNN Model Classification Report:")
print(classification_report(y_test, y_pred_rnn))

# Hyperparameter Optimization for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='roc_auc')
grid_search_rf.fit(X_train_scaled, y_train)
best_params_rf = grid_search_rf.best_params_

# Train a RandomForest model with optimized parameters
rf_model = RandomForestClassifier(**best_params_rf, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Save the RandomForest model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Predict with RandomForest
y_pred_rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = (y_pred_rf_prob > threshold).astype(int)

# Evaluate metrics for RandomForest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_prob)

# Print RandomForest evaluation metrics
print(f"RandomForest Model Accuracy: {accuracy_rf:.2f}")
print(f"RandomForest Model Precision: {precision_rf:.2f}")
print(f"RandomForest Model Recall: {recall_rf:.2f}")
print(f"RandomForest Model F1 Score: {f1_rf:.2f}")
print(f"RandomForest Model ROC AUC Score: {roc_auc_rf:.2f}")

# Display RandomForest classification report
print("\nRandomForest Model Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Ensemble predictions (average probabilities)
y_pred_ensemble_prob = (y_pred_prob_rnn + y_pred_rf_prob) / 2
y_pred_ensemble = (y_pred_ensemble_prob > threshold).astype(int)

# Calculate evaluation metrics for the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble_prob)

# Print evaluation metrics
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.2f}")
print(f"Ensemble Model Precision: {precision_ensemble:.2f}")
print(f"Ensemble Model Recall: {recall_ensemble:.2f}")
print(f"Ensemble Model F1 Score: {f1_ensemble:.2f}")
print(f"Ensemble Model ROC AUC Score: {roc_auc_ensemble:.2f}")

# Display classification report
print("\nEnsemble Model Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Plot all the visualizations in one window
fig, axs = plt.subplots(4, 2, figsize=(20, 20))

# Confusion Matrix for Ensemble model
conf_matrix = confusion_matrix(y_test, y_pred_ensemble)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'], ax=axs[0, 0])
axs[0, 0].set_title('Confusion Matrix - RNN + Random Forest Ensemble Model')
axs[0, 0].set_xlabel('Predicted')
axs[0, 0].set_ylabel('Actual')

# Confusion Matrix with Normalized Values
conf_matrix_norm = confusion_matrix(y_test, y_pred_ensemble, normalize='true')
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'], ax=axs[0, 1])
axs[0, 1].set_title('Normalized Confusion Matrix - RNN + Random Forest Ensemble Model')
axs[0, 1].set_xlabel('Predicted')
axs[0, 1].set_ylabel('Actual')

# ROC curve
fpr_rnn, tpr_rnn, _ = roc_curve(y_test, y_pred_prob_rnn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_prob)
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_pred_ensemble_prob)
axs[1, 0].plot(fpr_rnn, tpr_rnn, label='RNN ROC (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_prob_rnn)))
axs[1, 0].plot(fpr_rf, tpr_rf, label='RandomForest ROC (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_rf_prob)))
axs[1, 0].plot(fpr_ensemble, tpr_ensemble, label='Ensemble ROC (AUC = {:.2f})'.format(roc_auc_ensemble))
axs[1, 0].plot([0, 1], [0, 1], 'k--')
axs[1, 0].set_xlabel('False Positive Rate')
axs[1, 0].set_ylabel('True Positive Rate')
axs[1, 0].set_title('ROC Curve')
axs[1, 0].legend(loc='best')

# Precision-Recall curve
precision_rnn, recall_rnn, _ = precision_recall_curve(y_test, y_pred_prob_rnn)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf_prob)
precision_ensemble, recall_ensemble, _ = precision_recall_curve(y_test, y_pred_ensemble_prob)
axs[1, 1].plot(recall_rnn, precision_rnn, label='RNN PRC')
axs[1, 1].plot(recall_rf, precision_rf, label='RandomForest PRC')
axs[1, 1].plot(recall_ensemble, precision_ensemble, label='Ensemble PRC')
axs[1, 1].set_xlabel('Recall')
axs[1, 1].set_ylabel('Precision')
axs[1, 1].set_title('Precision-Recall Curve')
axs[1, 1].legend(loc='best')

# Learning curves for RNN model
axs[2, 0].plot(rnn_history.history['loss'], label='Training Loss')
axs[2, 0].plot(rnn_history.history['val_loss'], label='Validation Loss')
axs[2, 0].set_xlabel('Epochs')
axs[2, 0].set_ylabel('Loss')
axs[2, 0].set_title('RNN Training and Validation Loss')
axs[2, 0].legend(loc='best')

axs[2, 1].plot(rnn_history.history['accuracy'], label='Training Accuracy')
axs[2, 1].plot(rnn_history.history['val_accuracy'], label='Validation Accuracy')
axs[2, 1].set_xlabel('Epochs')
axs[2, 1].set_ylabel('Accuracy')
axs[2, 1].set_title('RNN Training and Validation Accuracy')
axs[2, 1].legend(loc='best')

# Histogram of Predicted Probabilities
axs[3, 0].hist(y_pred_prob_rnn, bins=50, alpha=0.5, label='RNN Predicted Probabilities')
axs[3, 0].hist(y_pred_rf_prob, bins=50, alpha=0.5, label='RandomForest Predicted Probabilities')
axs[3, 0].hist(y_pred_ensemble_prob, bins=50, alpha=0.5, label='Ensemble Predicted Probabilities')
axs[3, 0].set_xlabel('Predicted Probability')
axs[3, 0].set_ylabel('Frequency')
axs[3, 0].set_title('Histogram of Predicted Probabilities')
axs[3, 0].legend(loc='best')

# Feature Importances
rf_importances = rf_model.feature_importances_
sorted_idx = np.argsort(rf_importances)
axs[3, 1].barh(range(len(sorted_idx)), rf_importances[sorted_idx], align='center')
axs[3, 1].set_yticks(range(len(sorted_idx)))
axs[3, 1].set_yticklabels(X.columns[sorted_idx])
axs[3, 1].set_xlabel('Feature Importance')
axs[3, 1].set_title('RandomForest Feature Importances')

plt.tight_layout()
plt.show()
