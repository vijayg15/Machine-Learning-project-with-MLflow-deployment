# Machine-Learning-project-with-MLflow-deployment
A complete end to end machine learning project (Bank Customer Churn Prediction) with MLflow and MLOps.

## Bank Customer Churn Prediction
Customer Churn prediction means knowing which customers are likely to leave or unsubscribe from the service. Customer churn is important because it costs more to acquire new customers than to sell to existing customers. Following are the benefits of analyzing Customer Churn Prediction:
- Customer retention 
- Increase profits 
- Improve the customer experience 
- Optimize the product and services


Download the datasets from [here](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)

The project is divided into 5-main pipelines:
#### 1. Data Ingestion: 
In data ingestion pipeline, follow the following sequences: 
1. Change the *source_URL*, line 6 in [config.yaml](config/config.yaml) where the zip file of dataset is stored in your local machine.

#### 2. Data Validation:
In this pipeline, the following sequences are followed:
1. Change the last part in *unzip_data_dir*, line 13 in [config.yaml](config/config.yaml) to the file name (Churn_Modelling.csv).
2. Update all the column names, its data types and target_column name in [schema.yaml](schema.yaml) .

#### 3. Data Transformation:
In this pipeline, follow the below sequences: 
1. Perform EDA and fearure engineering in [03_data_transformation.ipynb](research/03_data_transformation.ipynb). 
2. Import all the required packages and libraries and modify the *fe_and_pre_process* function in *DataTransformation* class as per all feature engineering and preprocess steps in [data_transformation.py](src/mlProject/components/data_transformation.py)

#### 4. Model Trainer:
In this pipeline, follow the below sequences: 
1. Select the ML algorithm to be used. 
2. Insert all the hyperparameters in [params.yaml](params.yaml) for the selected model in step 1. 
3. Update all the related hyperparameters and its datatypes for the model selected in step 1 in class *ModelTrainerConfig* in [config_entity.py](src/mlProject/entity/config_entity.py). 
4. Add and update all the related hyperparameters in function *get_model_trainer_config* in [configuration.py](src/mlProject/config/configuration.py). 
5. Initialize and fit the model selected in step 1, in function *train* of class *ModelTrainer* in [model_trainer.py](src/mlProject/components/model_trainer.py) 

#### 5. Model Evaluation:
In this pipeline, follow the below sequences: 
1. Update the *mlflow_uri* in *get_model_evaluation_config* function in [configuration.py](src/mlProject/config/configuration.py).

Now, after setting up the [dagshub](https://dagshub.com/dashboard) account, edit the [main.py](main.py) file and update the following variables with yours: <br />
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  * os.environ["MLFLOW_TRACKING_URI"] <br />
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  * os.environ["MLFLOW_TRACKING_USERNAME"] <br />
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  * os.environ["MLFLOW_TRACKING_PASSWORD"]

Now, run the main file with 
```
    python main.py
```


