print("Starting...")

import joblib

try:
    pipe = joblib.load('best_model.pkl')
    print("Model loaded successfully!")
    print(pipe)
except Exception as e:
    print("Error loading model:", e)

print("Done.")
