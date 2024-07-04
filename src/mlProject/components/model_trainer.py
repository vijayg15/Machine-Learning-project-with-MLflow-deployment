import os
import joblib
import pandas as pd
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        #X_val = test_data.drop([self.config.target_column], axis=1)
        #y_val = test_data[[self.config.target_column]]


        clf = RandomForestClassifier(n_estimators = self.config.n_estimators,
                                     criterion = self.config.criterion,
                                     #max_depth = self.config.max_depth,
                                     min_samples_split = self.config.min_samples_split,
                                     min_samples_leaf = self.config.min_samples_leaf,
                                     bootstrap = self.config.bootstrap,
                                     #ccp_alpha = self.config.ccp_alpha,
                                     n_jobs=-1, verbose=1, random_state=40)
        clf.fit(X_train, y_train)

        joblib.dump(clf, os.path.join(self.config.root_dir, self.config.model_name))

