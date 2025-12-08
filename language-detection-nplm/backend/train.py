from model import train_model
import os

if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'nplm-model.pth')
    vec_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    print('Training model (this may take a while)...')
    model, vectorizer = train_model(epochs=30, save_model_path=model_path, save_vectorizer_path=vec_path)
    print('Model trained and saved to', model_path)
