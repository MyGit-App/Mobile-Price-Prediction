import pickle
import numpy as np

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Loaded scaler and model successfully')
except Exception as e:
    print('Error loading files:', e)
    raise

print('scaler n_features_in_', getattr(scaler, 'n_features_in_', None))

# Sample input matching FEATURE_ORDER:
# [Sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, RearCam, Front_Cam, battery, thickness]
features = np.array([[600, 170, 5.2, 335, 4, 1.5, 32, 2, 12, 5, 2800, 8.9]])
print('features shape', features.shape)

try:
    features_scaled = scaler.transform(features)
    print('features_scaled shape', features_scaled.shape)
    pred = model.predict(features_scaled)
    print('prediction:', pred)
except Exception as e:
    print('Error during transform/predict:', e)
    raise
