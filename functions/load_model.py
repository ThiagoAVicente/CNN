import pickle

def load():
    # Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("Model loaded successfully.")
    return model
