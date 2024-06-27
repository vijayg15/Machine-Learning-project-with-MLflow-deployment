import os
from mlProject import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from mlProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def fe_and_pre_process(self):
        data = pd.read_csv(self.config.data_path)

        df = data.dropna()
        df.drop_duplicates(inplace=True)

        df_encoded = pd.get_dummies(df, columns = ['Gender', 'Geography'], drop_first = True)

        df_encoded['HasCrCard'] = df_encoded['HasCrCard'].astype(np.uint8)
        df_encoded['IsActiveMember'] = df_encoded['IsActiveMember'].astype(np.uint8)
        
        df_encoded['normalizedCreditScore'] = StandardScaler().fit_transform(df_encoded.CreditScore.values.reshape(-1,1))
        df_encoded['normalizedAge'] = StandardScaler().fit_transform(df_encoded.Age.values.reshape(-1,1))
        df_encoded['normalizedTenure'] = StandardScaler().fit_transform(df_encoded.Tenure.values.reshape(-1,1))
        df_encoded['normalizedBalance'] = StandardScaler().fit_transform(df_encoded.Balance.values.reshape(-1,1))
        df_encoded['normalizedEstimatedSalary'] = StandardScaler().fit_transform(df_encoded.EstimatedSalary.values.reshape(-1,1))

        df_scaled = df_encoded.drop(["RowNumber", "CustomerId", "Surname", "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"], axis = 1)
        
        X = df_scaled.drop(['Exited'], axis='columns')
        y = df_scaled.Exited

        X_resample, y_resample = SMOTE().fit_resample(X, y)

        data = pd.concat([X_resample, y_resample], axis=1)

        # Split the data into training and test sets. (0.80, 0.20) split.
        train, test = train_test_split(data, test_size=0.20, random_state=12)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Feature Engineering and Pre-processing is done!")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        