from collections import namedtuple
from enum import Enum
from importlib import import_module

from print import pr_red

class Classifier(Enum):
    SVC = "SVC"
    KNN = "KNN"
    RF = "Random Forest Classifier"
    LR = "Logist Regression"
    PPIIBM_first_item = "Pair Prediction by Item Identification Baseline Model (first item mode)",
    PPIIBM_both_items = "Pair Prediction by Item Identification Baseline Model (both items mode)"
    
Model = namedtuple('Model', 'clf param_grid name')

def _prepare_models(models_to_exec, use_GPU, model_configurations):
    models = []
    # Iterate over the models to execute
    for model_name in models_to_exec:
        try:
            config = model_configurations[model_name]
        except KeyError:
            pr_red(f"No configuration found for model {model_name}")
            continue

        param_grid = config['param_grid']
        classifier_config = config['cuML'] if use_GPU else config['sklearn']

        module_name = classifier_config['module']
        class_name = classifier_config['class']
        try:
            module = import_module(module_name)
        except ImportError as e:
            pr_red(f"Failed to import module {module_name}: {e}")
            continue
        except AttributeError as e:
            pr_red(f"Failed to get class {class_name} from module {module_name}: {e}")
            continue

        classifier_instance = getattr(module, classifier_config['class'])(**classifier_config['params'])

        gpu_or_cpu = "GPU" if use_GPU else "CPU"
        print(f"Using {model_name} Classifier for {gpu_or_cpu}")
        models.append(Model(classifier_instance, param_grid, model_name))
    return models

def prepare_models(models_to_exec, random_state, use_GPU, print_debug_messages=False):
    # Define a dictionary with the configurations of the models
    model_configurations = {
        # Nested CV: kNN + GridSearchCV (baish-line)
        Classifier.KNN.name: {
            'param_grid': {'n_neighbors': [25, 75, 125]},
            'cuML': {'module': 'models.modelsWrapper', 'class': 'cuMLKNNWrapper', 'params': {}},
            'sklearn': {'module': 'sklearn.neighbors', 'class': 'KNeighborsClassifier', 'params': {}}
        },
        # Nested CV: Logistic regression pipeline + GridSearchCV
        Classifier.LR.name: {
            'param_grid': {'C': [0.0001, 1, 10], 'penalty': ['l1', 'l2']},
            'cuML': {'module': 'cuml.linear_model', 'class': 'LogisticRegression', 'params': {'max_iter': 1000, 'solver': 'qn'}},
            'sklearn': {'module': 'sklearn.linear_model', 'class': 'LogisticRegression', 'params': {'max_iter': 1000, 'solver': 'liblinear'}}
        },
        # Nested CV: Random Forest + GridSearchCV
        Classifier.RF.name: {
            'param_grid': {'n_estimators': [100, 200], 'min_samples_leaf': [1, 10, 50], 'max_samples': [0.75, 1.0]},
            'cuML': {'module': 'cuml.ensemble', 'class': 'RandomForestClassifier', 'params': {'random_state': random_state, 'n_streams': 1}},
            'sklearn': {'module': 'sklearn.ensemble', 'class': 'RandomForestClassifier', 'params': {'random_state': random_state}}
        },
        # Nested CV: SVC + GridSearchCV
        # https://stackabuse.com/understanding-svm-hyperparameters/
        Classifier.SVC.name: {
            'param_grid': {'kernel': ['linear', 'rbf'], 'C': [0.0001, 1, 10], 'gamma': [1, 10, 100]},
            'cuML': {'module': 'cuml.svm', 'class': 'SVC', 'params': {'random_state': random_state}},
            'sklearn': {'module': 'sklearn.svm', 'class': 'SVC', 'params': {'random_state': random_state}}
        },
        Classifier.PPIIBM_first_item.name: {
            'param_grid': {},
            'cuML': {'module': 'models.ppiibm', 'class': 'PPIIBM', 'params': {'print_debug_messages': print_debug_messages}},
            'sklearn': {'module': 'models.ppiibm', 'class': 'PPIIBM', 'params': {'print_debug_messages': print_debug_messages}}
        },
        Classifier.PPIIBM_both_items.name: {
            'param_grid': {},
            'cuML': {'module': 'models.ppiibm', 'class': 'PPIIBM', 'params': {'singleItemMode': False, 'print_debug_messages': print_debug_messages}},
            'sklearn': {'module': 'models.ppiibm', 'class': 'PPIIBM', 'params': {'singleItemMode': False,'print_debug_messages': print_debug_messages}}
        }
    }

    return _prepare_models(models_to_exec, use_GPU, model_configurations)