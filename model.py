import pickle
from config import MODEL_PATH


class Model:
    def __init__(self):
        self.model = None

    def load(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

        except Exception as e:
            print(f"Error loading model {e}")

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model is not loaded")
        return self.model.predict(features)



model = Model()
