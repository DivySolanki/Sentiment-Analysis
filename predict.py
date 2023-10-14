import sys
import pandas as pd
import pickle

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model_Ridge.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            with open(model_path, 'rb') as file_obj:
                model = pickle.load(file_obj)
            with open(preprocessor_path, 'rb') as file_obj:
                preprocessor = pickle.load(file_obj)
            data_scaled = preprocessor.transform(features['text'])
            preds  = model.predict(data_scaled)
            sentiment = ''
            if preds[0] >= 0.70:
                sentiment = 'Positive'
            else:
                sentiment = 'Negative'
            return sentiment
        
        except Exception as e:
            raise e

class CustomData:
    def __init__(self,
                 text: str,
                ) -> None:
        self.text = text

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "text": [self.text]
            }

            return pd.DataFrame(custom_data_input_dict, columns = ['text'])

        except Exception as e:
            raise e