import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, Counter
import time
import joblib
import os
import glob
import re
import serial
import shutil

# for Target Applications our group chose " Distraction Lock – browser tab auto-mutes YouTube when focus drops."
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# sliding window with length 8 for our ML classifier, as req.
WINDOW_SIZE = 8
MODEL_FILENAME = 'attention_model.h5'
TFLITE_MODEL_FILENAME = 'attention_model.tflite'
SCALER_FILENAME = 'data_scaler.joblib'
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=h0e2HAPTGF4&t=441s"
ARDUINO_SERIAL_PORT = '/dev/cu.usbserial-10' 
ARDUINO_BAUD_RATE = 57600

def load_and_prepare_data():
    print("\n--- Step 1: Loading and Preparing Data from All Participants ---")
    relaxed_files, concentrating_files = glob.glob("relaxed_*.txt"), glob.glob("concentrating_*.txt")
    if not relaxed_files or not concentrating_files:
        raise FileNotFoundError("Data files not found! Please run data_collector.py first.")
    
    relaxed_scores = np.concatenate([np.loadtxt(f, dtype=int) for f in relaxed_files])
    concentrating_scores = np.concatenate([np.loadtxt(f, dtype=int) for f in concentrating_files])
    
    print(f"Found data from {len(relaxed_files)} participant(s). Total samples: {len(relaxed_scores) + len(concentrating_scores)}")

    X_raw = np.concatenate((relaxed_scores, concentrating_scores))
    y_raw = np.concatenate((np.zeros_like(relaxed_scores), np.ones_like(concentrating_scores)))
    
    X, y = [], []
    for i in range(len(X_raw) - WINDOW_SIZE):
        X.append(X_raw[i : i + WINDOW_SIZE])
        y.append(y_raw[i + WINDOW_SIZE - 1])
    X, y = np.array(X), np.array(y)
    
    # split train, test into ratio 8:2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # normalize the attention data within [0,1]
    scaler = MinMaxScaler()
    # scale the data
    scaler.fit(X_train.reshape(-1, 1))
    X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Data scaler saved to {SCALER_FILENAME}")

    X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    print("\n--- Step 2: Building and Training AI Model ---")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    print("\nTraining model...")
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping], verbose=1)
    model.save(MODEL_FILENAME)
    print(f"\nTrained model saved to {MODEL_FILENAME}")
    
    print("\n--- Visualizing Training History ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
   
    print("\n--- Model Performance Report ---")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Accuracy on Hold-out Data: {accuracy * 100:.2f}%")
    if accuracy < 0.85: print("WARNING: Model accuracy is below the 85% project goal.")
    
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Relaxed', 'Concentrating']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Relaxed', 'Concentrating'], yticklabels=['Relaxed', 'Concentrating'])
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()
    return model

def convert_to_tflite(model):
    print("\n--- Step 3: Converting to TensorFlow Lite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_FILENAME, 'wb') as f:
        f.write(tflite_model)
    print(f"Optimized TFLite model saved to {TFLITE_MODEL_FILENAME}")

def run_live_distraction_lock():
    print("\n--- Step 4: Launching LIVE 'Distraction Lock' Demonstration ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILENAME)
        interpreter.allocate_tensors()
        scaler = joblib.load(SCALER_FILENAME)
        input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    except FileNotFoundError as e:
        print(f"ERROR: A required file is missing: {e}. Please run the training steps first.")
        return
    
    profile_path = os.path.expanduser("~/Desktop/chrome_profile_for_neuro_ai")
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path)
    
    # 2. Configure Chrome options with the "stealth" flag.
    options = webdriver.ChromeOptions()
    
    # This is the "Silver Bullet". It tells Chrome to hide the automation flags.
    options.add_argument("--disable-blink-features=AutomationControlled") 

    # These flags create a stable, sandboxed environment.
    options.add_argument(f"--user-data-dir={profile_path}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    print("Setting up Chrome browser in stealth mode...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    print(f"Navigating to YouTube: {YOUTUBE_VIDEO_URL}")
    driver.get(YOUTUBE_VIDEO_URL)
    
    try:
        print("Waiting for YouTube video player to be ready...")
        video_player_element = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "movie_player"))
        )
        print("Player is ready.")
    except Exception as e:
        print(f"FATAL ERROR: Could not find the YouTube player on the page. {e}")
        driver.quit()
        return

    history_buffer = deque(maxlen=WINDOW_SIZE)
    is_muted = False
    
    prediction_history = deque(maxlen=5) 
   
    print(f"\n--- Attempting to Connect to Arduino on {ARDUINO_SERIAL_PORT}... ---")
    try:
        with serial.Serial(ARDUINO_SERIAL_PORT, ARDUINO_BAUD_RATE, timeout=1) as ser:
            print("Successfully connected to Arduino. Starting Live Control Loop...")
            print("(Press Ctrl+C in this terminal to stop the demo)")
            while True:
                current_focus = -1
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if "Attation" in line:
                    signal_match = re.search(r'SignalQuality: (\d+)', line)
                    attention_match = re.search(r'Attation: (\d+)', line) 
                    
                    if signal_match and attention_match and int(signal_match.group(1)) == 0:
                        current_focus = int(attention_match.group(1))
                
                if current_focus != -1:
                    history_buffer.append(current_focus)
                    status_line = f"Live Score: {current_focus:3d}"
                    if len(history_buffer) == WINDOW_SIZE:
                        input_array = np.array(history_buffer, dtype=np.float32)
                        scaled_input = scaler.transform(input_array.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
                        
                        interpreter.set_tensor(input_details[0]['index'], scaled_input)
                        # run the model
                        interpreter.invoke()
                        prediction_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
                        
                        current_state = "DISTRACTED" if prediction_score < 0.5 else "FOCUSED"
                        prediction_history.append(current_state)
                        
                        if len(prediction_history) == prediction_history.maxlen:
                            # 3. Get the most common state (majority vote)
                            stable_state = Counter(prediction_history).most_common(1)[0][0]
                            
                            # 4. Perform actions based on the STABLE state
                            if stable_state == "DISTRACTED":
                                if not is_muted:
                                    print(f"{status_line} -> STATE STABILIZED: DISTRACTED | ACTION: Muting YouTube!")
                                    driver.execute_script("document.querySelector('video').muted = true;")
                                    is_muted = True
                                else:
                                    print(f"{status_line} -> STATE STABILIZED: DISTRACTED | STATUS: Already muted.")
                            else: # Stable state is FOCUSED
                                if is_muted:
                                    print(f"{status_line} -> STATE STABILIZED: FOCUSED    | ACTION: Unmuting YouTube!")
                                    driver.execute_script("document.querySelector('video').muted = false;")
                                    is_muted = False
                                else:
                                    print(f"{status_line} -> STATE STABILIZED: FOCUSED    | STATUS: Already unmuted.")
                    else:
                        print(f"Gathering initial data ({len(history_buffer)}/{WINDOW_SIZE})...")
                
    except serial.SerialException:
        print(f"\nFATAL ERROR: Could not open serial port '{ARDUINO_SERIAL_PORT}'.")
        print("TROUBLESHOOTING: Check if Arduino is plugged in, port is correct, and Serial Monitor is closed.")
    except Exception as e:
        print(f"\nAn error occurred during the live demo: {e}")
    finally:
        print("\nShutting down browser...")
        driver.quit()

def main():
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(SCALER_FILENAME):
        print("Model or scaler not found. Starting full training pipeline...")
        X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data()
        model = build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        convert_to_tflite(model)
    else:
        print("Model artifacts already exist. Skipping training and jumping to live demo.")
    run_live_distraction_lock()

if __name__ == '__main__':
    main()