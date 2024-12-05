import pickle

def save( model ):
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("Model saved successfully.")
