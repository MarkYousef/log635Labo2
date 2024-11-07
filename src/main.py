import pickle
import time
import perceptron

def open_file(file_name):
    
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def get_dataset(name):
   
    data_files = {
        'training': ('listeImages_entrainement', 'listeLabels_entrainement'),
        'validation': ('listeImages_validation', 'listeLabels_validation'),
        'test': ('listeImages_test', 'listeLabels_test')
    }
    
    if name not in data_files:
        raise ValueError("Nom de dataset invalide. Choisissez 'training', 'validation' ou 'test'.")

    X_name, y_name = data_files[name]
    X = (open_file(X_name))
    Y = (open_file(y_name))
    
    return X, Y

def train(): 
    
    X_train, y_train = get_dataset('training')
    #X_val, y_val = get_dataset('validation')

    perceptron_model = perceptron.Perceptron()

    start_train = time.time()
    perceptron_model.train(X_train, y_train)
    end_train = time.time()
    train_duration = end_train - start_train
    print(f"Training time: {train_duration:.2f} seconds")

    # Mesure du temps de prédiction

    # Évaluation des performances
  