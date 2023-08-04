from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearnex import patch_sklearn
from dataset import load_keel_file
from itertools import product
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pandas as pd
import itertools
import sys


def load_dataset(path):

    # Load dataset in keel format
    dataset = load_keel_file(path)

    # Get inputs and target variables
    X, y = dataset.get_data()

    # Encoder target variable
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


def custom_auc(y_true, y_pred):
    lb = LabelBinarizer()

    y_true_lb = lb.fit_transform(y_true)
    y_pred_lb = lb.transform(y_pred)

    return roc_auc_score(y_true_lb, y_pred_lb)


def run(kwargs, n_processors):

    data_path = kwargs.get('dataset')
    random_seed = kwargs.get('seed')
    folds = kwargs.get('n_fold')
    repeats = kwargs.get('n_repeat')

    try:
        # Load dataset
        dataset = load_keel_file(data_path)

        # Get inputs and target variables
        X, y = dataset.get_data()

        # Normalize inputs variables
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Encoder target variable
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        parameters = {
            'hidden_layer_sizes': list(itertools.chain.from_iterable([list(product([100, 150, 200], repeat=layer)) for layer in [1, 2, 3]])),
            'activation': ['tanh', 'relu'],
            'learning_rate_init': [0.1, 0.01, 0.001]
        }

        # Stratify cross validation
        kfold = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_seed)

        # GridSearch
        grid_search = GridSearchCV(MLPClassifier(learning_rate='adaptive', max_iter=500), parameters, scoring=make_scorer(custom_auc), cv=kfold, verbose=10, n_jobs=n_processors)
        grid_search.fit(X, y)

        # Results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(f'./results/{dataset.name}.csv', index=False)
    except Exception as e:
        print(f'Erro ao processar dataset {data_path}.\nErro: {e}')


if __name__ == "__main__":

    patch_sklearn()

    n_fold = 10
    n_repeat = 10
    seed = 491994

    # number of processors in CPU
    try: 
        n_processors = int(sys.argv[1])
    except:
        n_processors = -1

    # List all dataset
    datasets_path = list(glob('./data/*.dat'))
    datasets_path.sort()

    # Results path
    results_path = Path('./results')
    results_path.mkdir(exist_ok=True)
    
    already_results = list(results_path.glob('*.csv'))

    args = []
    for i, dataset_path in enumerate(datasets_path):

        if f'./results/{Path(dataset_path).name[:-4]}.csv' in already_results:
            continue

        arg = {}
        arg['dataset'] = dataset_path
        arg['seed'] = seed
        arg['n_fold'] = n_fold
        arg['n_repeat'] = n_repeat
        
        # Run eval
        run(arg, n_processors)

        
