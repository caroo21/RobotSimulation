import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.models import Model

def load_and_analyze_data(filename="kuka_trajectories.pkl"):
    """Daten laden und analysieren"""
    with open(filename, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Datensatz geladen: {len(trajectories)} Trajektorien")
    
    return trajectories

def extract_state_features(state):
    """State in Feature-Vektor umwandeln"""
    features = []
    
    # Roboter Joint Positions (7 Werte)
    features.extend(state['joint_positions'])
    
    # End-Effektor Position (3 Werte)  
    features.extend(state['end_effector_pos'])
    
    # Würfel Position (3 Werte)
    features.extend(state['cube_position'])
    
    # Zylinder Position (3 Werte)
    features.extend(state['cylinder_position'])
    
    # Gesamt: 7 + 3 + 3 + 3 = 16 Features
    return features

def prepare_tensorflow_data(trajectories):
    """Daten für TensorFlow vorbereiten"""
    states = []
    actions = []
    
    for traj_info in trajectories:
        trajectory = traj_info['trajectory']
        
        for step_data in trajectory:
            state = step_data['state']
            action = step_data['action']
            
            # Nur Schritte mit target_position verwenden
            if 'target_position' in action and action['target_position']:
                # State features extrahieren
                state_vector = extract_state_features(state)
                target_pos = action['target_position']
                
                states.append(state_vector)
                actions.append(target_pos)
    
    # Zu NumPy Arrays konvertieren
    X = np.array(states, dtype=np.float32)
    y = np.array(actions, dtype=np.float32)
    
    print(f"Dataset erstellt: {len(X)} Trainingssamples")
    print(f"State Shape: {X.shape}")
    print(f"Action Shape: {y.shape}")
    
    return X, y

def create_tensorflow_model(state_dim=16, action_dim=3, hidden_dim=128):
    """TensorFlow Model erstellen"""
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim, activation='relu', input_shape=(state_dim,)),
        keras.layers.Dense(hidden_dim, activation='relu'),
        keras.layers.Dense(hidden_dim, activation='relu'),
        keras.layers.Dense(action_dim) # Output: [x, y, z] Position keine Aktivierung für lineare Ausgabe
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_advanced_robot_model(state_dim, action_dim, hidden_dim=128):
    # Input
    inputs = Input(shape=(state_dim,), name='robot_state')
    
    # Erste Verarbeitungsschicht
    x1 = Dense(hidden_dim, activation='relu', name='perception')(inputs)
    x1 = Dropout(0.2)(x1)
    
    # Zweite Verarbeitungsschicht  
    x2 = Dense(hidden_dim, activation='relu', name='reasoning')(x1)
    x2 = Dropout(0.2)(x2)
    
    # Skip Connection: Verbinde Input mit reasoning
    x2_with_skip = concatenate([x2, inputs], name='memory_connection')
    
    # Finale Entscheidungsschicht
    x3 = Dense(hidden_dim // 2, activation='relu', name='decision')(x2_with_skip)
    
    # Ausgabe
    outputs = Dense(action_dim, name='action')(x3)
    
    # Model zusammenbauen
    model = Model(inputs=inputs, outputs=outputs, name='SmartRobot')
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model

def train_tensorflow_model(model, X, y, epochs=100, validation_split=0.2):
    """Model trainieren"""
    
    # Callbacks für besseres Training
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    # Training
    history = model.fit(
        X, y,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Training Verlauf plotten"""
    
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_model_predictions(model, X, y, num_samples=5):
    """Model Vorhersagen testen"""
    # Zufällige Samples auswählen
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    print("\nModel Test - Vorhersagen vs. Ground Truth:")
    print("=" * 60)
    
    for i, idx in enumerate(indices):
        state = X[idx:idx+1]  # Batch dimension
        true_action = y[idx]
        predicted_action = model.predict(state, verbose=0)[0]
        
        error = np.linalg.norm(predicted_action - true_action)
        
        print(f"\nSample {i+1}:")
        print(f"True Action:      [{true_action[0]:.3f}, {true_action[1]:.3f}, {true_action[2]:.3f}]")
        print(f"Predicted Action: [{predicted_action[0]:.3f}, {predicted_action[1]:.3f}, {predicted_action[2]:.3f}]")
        print(f"Error (L2 norm):  {error:.3f}")

# HAUPTPROGRAMM
if __name__ == "__main__":
    print("Pick-and-Place Imitation Learning mit TensorFlow")
    print("=" * 60)
    
    # 1. Daten laden
    trajectories = load_and_analyze_data("kuka_trajectories.pkl")
    
    # 2. Daten vorbereiten
    X, y = prepare_tensorflow_data(trajectories)
    
    if len(X) == 0:
        print("Keine verwendbaren Daten gefunden!")
        print("Stelle sicher, dass die Trajektorien target_position enthalten.")
        exit()
    
    # 3. Model erstellen
    #model = create_tensorflow_model(state_dim=X.shape[1], action_dim=y.shape[1])
    model = create_advanced_robot_model(state_dim=X.shape[1], action_dim=y.shape[1])

    print(f"\nModel Architektur:")
    model.summary()
    
    # 4. Training
    print(f"\nStarte Training...")
    history = train_tensorflow_model(model, X, y, epochs=100)
    
    # 5. Model testen
    test_model_predictions(model, X, y)
    
    # 6. Model speichern
    model.save('pickplace_policy_tensorflow.h5')
    print(f"\nModel gespeichert: pickplace_policy_tensorflow.h5")
    
    # 7. Finale Statistiken
    final_loss = history.history['val_loss'][-1]
    final_mae = history.history['val_mae'][-1]
        
    # 8. Ergebnisse anzeigen
    plot_training_history(history)
    
    print(f"\nTraining abgeschlossen!")
    print(f"Finale Validation Loss: {final_loss:.6f}")
    print(f"Finale Validation MAE:  {final_mae:.6f}")
    print(f"Durchschnittlicher Positionsfehler: {final_mae*100:.1f} cm")
