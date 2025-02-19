import pickle

def save( model ):
    """saves the model in a file using pickle"""
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("Model saved successfully.")
