import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TemporalPoseTrainer:
    def __init__(self, dataset_path, sequence_length=10):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.features_per_frame = 34

    def build_lstm_network(self):
        """
        Creates a Recurrent Neural Network (LSTM) to analyze motion over time.
        Input shape: (time_steps, features_per_step) -> (10, 34)
        """
        model = Sequential([
            # LSTM Layer 1: Learns temporal dynamics and velocity of movements
            LSTM(64, return_sequences=False, input_shape=(self.sequence_length, self.features_per_frame)),
            Dropout(0.3),

            # Dense layers for final classification reasoning
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),

            # Binary Output (0 = fight, 1 = normal)
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_and_evaluate(self, model_save_path):
        print(f"[INFO] Loading sequence dataset from: {self.dataset_path}...")
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print(f"[ERROR] Dataset not found. Please run collection first.")
            return

        encoder = LabelEncoder()
        df['label_encoded'] = encoder.fit_transform(df['label'])

        # Extract features and convert to 3D Tensor for LSTM
        X_flat = df.drop(['label', 'label_encoded'], axis=1).values.astype(np.float32)
        y = df['label_encoded'].values.astype(np.float32)

        # CRITICAL: Reshape from 2D (Samples, 340) to 3D (Samples, 10, 34)
        # This gives the network the dimension of TIME.
        X = X_flat.reshape(-1, self.sequence_length, self.features_per_frame)

        # 70/15/15 Data Split
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.50, random_state=42, stratify=y_rem)

        print(f"[DATA SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        model = self.build_lstm_network()
        print("[INFO] LSTM Architecture compiled successfully.")
        model.summary()

        epochs = 100
        batch_size = 64
        print(f"[START] Training Temporal LSTM for {epochs} epochs...")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        print("\n[INFO] Running the ultimate evaluation on unseen Test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"[FINAL TEST RESULTS] Isolated Test Accuracy: {test_acc * 100:.2f}%")
        print(f"[FINAL TEST RESULTS] Isolated Test Loss: {test_loss:.4f}")

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"\n[SUCCESS] LSTM Temporal model saved to: {model_save_path}")


if __name__ == "__main__":
    dataset_file = "data/pose_dataset.csv"
    saved_model_file = "data/threat_model_lstm.keras"

    trainer = TemporalPoseTrainer(dataset_file)
    trainer.train_and_evaluate(saved_model_file)