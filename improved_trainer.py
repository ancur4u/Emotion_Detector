# Improved Emotion Model Trainer with Better Accuracy
# Addresses low confidence and wrong predictions

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import sys
from datetime import datetime
import cv2

def enhanced_feature_extraction(img):
    """Enhanced feature extraction for better emotion recognition"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    
    features = []
    
    # 1. Raw pixels (normalized)
    pixels = img.flatten() / 255.0
    features.extend(pixels)
    
    # 2. Facial regions focus (eyes, mouth, forehead)
    h, w = img.shape
    
    # Eye regions (upper 1/3)
    eye_region = img[:h//3, :]
    eye_features = eye_region.flatten() / 255.0
    eye_mean = np.mean(eye_features)
    eye_std = np.std(eye_features)
    features.extend([eye_mean, eye_std])
    
    # Mouth region (lower 1/3)
    mouth_region = img[2*h//3:, :]
    mouth_features = mouth_region.flatten() / 255.0
    mouth_mean = np.mean(mouth_features)
    mouth_std = np.std(mouth_features)
    features.extend([mouth_mean, mouth_std])
    
    # 3. Edge detection features
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    features.append(edge_density)
    
    # 4. Gradient features
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_mean = np.mean(gradient_magnitude)
    grad_std = np.std(gradient_magnitude)
    features.extend([grad_mean, grad_std])
    
    # 5. Texture features (simplified LBP)
    lbp_hist = compute_lbp_histogram(img)
    features.extend(lbp_hist)
    
    return np.array(features)

def compute_lbp_histogram(img, radius=1, n_points=8):
    """Compute Local Binary Pattern histogram"""
    h, w = img.shape
    lbp = np.zeros((h-2*radius, w-2*radius))
    
    for i in range(radius, h-radius):
        for j in range(radius, w-radius):
            center = img[i, j]
            pattern = 0
            
            # Sample points in circle
            for p in range(n_points):
                angle = 2 * np.pi * p / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                
                if 0 <= x < h and 0 <= y < w:
                    pattern |= (img[x, y] >= center) << p
            
            lbp[i-radius, j-radius] = pattern
    
    # Create histogram
    hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 2**n_points))
    return hist / np.sum(hist)  # Normalize

def load_and_preprocess_dataset(csv_path, sample_size=None):
    """Load dataset with enhanced preprocessing"""
    print(f"📂 Loading dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Sample dataset if requested (for faster training)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"📉 Sampled to {sample_size} examples")
    
    # Extract features and labels
    X = []
    y = []
    
    print("🔄 Extracting enhanced features...")
    for idx, row in df.iterrows():
        try:
            # Parse pixels
            pixels = np.array([int(x) for x in row['pixels'].split(' ')])
            img = pixels.reshape(48, 48).astype('uint8')
            
            # Enhanced feature extraction
            features = enhanced_feature_extraction(img)
            
            X.append(features)
            y.append(row['emotion'])
            
            if len(X) % 1000 == 0:
                print(f"  Processed: {len(X)}/{len(df)}")
                
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Feature extraction completed: {X.shape}")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print("📊 Class distribution:")
    for emotion_id, count in zip(unique, counts):
        if emotion_id < len(emotion_names):
            print(f"   {emotion_names[emotion_id]}: {count}")
    
    return X, y

def train_improved_model(X, y, model_type='GradientBoosting'):
    """Train improved model with better hyperparameters"""
    print(f"\n🚀 Training improved {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"📊 Computed class weights: {class_weight_dict}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=150, random_state=42)  # Increased components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"✅ PCA applied: {X_train_pca.shape}")
    print(f"📊 Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create and train model with better parameters
    if model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=1
        )
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=200,  # Increased
            max_depth=25,      # Increased
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle imbalance
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'SVM':
        model = SVC(
            kernel='rbf',
            C=10.0,           # Increased regularization
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation before final training
    print("🔄 Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=3, scoring='accuracy')
    print(f"📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    print("🔄 Training final model...")
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Final Model Performance:")
    print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get prediction probabilities to check confidence
    y_pred_proba = model.predict_proba(X_test_pca)
    avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
    print(f"📊 Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    # Detailed report
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotion_names))
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"improved_emotion_model_{model_type.lower()}_{timestamp}.pkl"
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'model_type': model_type,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'emotion_names': emotion_names,
        'feature_method': 'enhanced',
        'class_weights': class_weight_dict
    }
    
    joblib.dump(model_data, model_filename)
    print(f"💾 Model saved as: {model_filename}")
    
    return model_data, accuracy, avg_confidence

def main():
    print("🎭 Improved Emotion Recognition Model Trainer")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python improved_trainer.py <path_to_csv> [sample_size]")
        print("Example: python improved_trainer.py fer2013_converted.csv 10000")
        return
    
    csv_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    # Load dataset
    X, y = load_and_preprocess_dataset(csv_path, sample_size)
    if X is None:
        return
    
    # Get model choice
    print("\n🔧 Available models:")
    print("1. GradientBoosting (Recommended - Best accuracy)")
    print("2. RandomForest (Fast training)")
    print("3. SVM (Good for small datasets)")
    
    choice = input("Choose model (1-3) [1]: ").strip()
    
    model_map = {
        '1': 'GradientBoosting',
        '2': 'RandomForest', 
        '3': 'SVM',
        '': 'GradientBoosting'
    }
    
    model_type = model_map.get(choice, 'GradientBoosting')
    
    print(f"\n📋 Training Configuration:")
    print(f"   Model: {model_type}")
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Enhanced features: ✅")
    print(f"   Class balancing: ✅")
    
    # Train model
    try:
        model_data, accuracy, avg_confidence = train_improved_model(X, y, model_type)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Average confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        if avg_confidence > 0.7:
            print("✅ High confidence model - should work well!")
        elif avg_confidence > 0.5:
            print("⚠️ Moderate confidence - consider more training data")
        else:
            print("❌ Low confidence - model may need improvement")
        
        print(f"\n📝 Next Steps:")
        print(f"1. Copy the .pkl file to your Streamlit app directory")
        print(f"2. Rename it to 'emotion_model.pkl' or load it directly")
        print(f"3. Test with the improved model!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()