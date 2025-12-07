from utils.preprocessing import normalize_x_y_pressure, calculate_metrics
from .signature_verifier import SignatureVerifier
import numpy as np
import pandas as pd
import math
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler

class KolmogorovSmirnovSignatureVerifier(SignatureVerifier):

   def __init__(self, genuine_signature: pd.DataFrame, significance_level=0.01):
       self.significance_level = significance_level
       self.scaler = StandardScaler()
       self.genuine_features = self.preprocess_signature(genuine_signature)

   def preprocess_signature(self, signature: pd.DataFrame, n_points=100) -> pd.DataFrame:
       needed_metrics = {'Speed', 'Acceleration', 'Curvature', 'PressureVelocity'}
       signature = calculate_metrics(signature.copy(), needed_metrics)

       features_to_normalize = ['X', 'Y', 'Pressure', 'Speed', 'Acceleration', 'Curvature', 'PressureVelocity']
       scaler = StandardScaler()
       
       for col in features_to_normalize:
           if col in signature.columns:
               data = signature[col].values.reshape(-1, 1)
               if len(data) > 1 and np.std(data) > 0:
                   signature[col] = scaler.fit_transform(data).flatten()
               else:
                   signature[col] = 0.0
           else:
               signature[col] = 0.0

       return signature

   def calculate_similarity_score(self, distances):
       """
       Превращает среднее расстояние KS (от 0 до 1) в вероятность (проценты).
       Эмпирически: 
         - расстояние < 0.15 -> высокая вероятность подлинности (>80%)
         - расстояние > 0.25 -> низкая вероятность (<40%)
       """
       if not distances:
           return 0.0
       
       avg_dist = np.mean(distances)
       k = 15
       threshold = 0.20 
       
       try:
           x = k * (avg_dist - threshold)
           prob = 1 / (1 + np.exp(x))
           return prob # Возвращает вероятность, что это Genuine (0..1)
       except:
           return 0.0

   def verify(self, test_signature: pd.DataFrame):
       if self.genuine_features is None:
           raise ValueError('Нет данных эталона')
      
       test_features = self.preprocess_signature(test_signature)
      
       features = ['X', 'Y', 'Btn', 'Curvature', 'Speed', 'Acceleration', 'PressureVelocity']
       
       weights = {
           'X': 1.0, 'Y': 1.0, 'Btn': 0.5, 
           'Curvature': 2.0, 'Speed': 3.0, 'Acceleration': 3.0, 'PressureVelocity': 1.5
       }
       
       ks_distances = []
       valid_weights = []

       for feature in features:
           if feature not in test_features.columns or feature not in self.genuine_features.columns:
               continue
               
           gen_data = self.genuine_features[feature]
           test_data = test_features[feature]
           
           if np.all(gen_data == 0) or np.all(test_data == 0):
               continue

           try:
               d_stat, p_val = ks_2samp(gen_data, test_data)
               w = weights.get(feature, 1.0)
               ks_distances.append(d_stat) 
               valid_weights.append(w)
               
           except Exception:
               pass

       if not ks_distances:
           return {
               'combined_p': 0, 'is_genuine': False, 
               'probability_forgery2': 0, 'details': "No valid features"
           }

       # Взвешенное среднее расстояние
       avg_distance = np.average(ks_distances, weights=valid_weights)
       
       # Превращаем расстояние в вероятность
       probability_genuine = self.calculate_similarity_score([avg_distance])
       
       # Вердикт: если вероятность > 50%
       is_genuine_verdict = probability_genuine > 0.5

       return {
           'combined_p': avg_distance, 
           'is_genuine': is_genuine_verdict,
           'probability_forgery2': probability_genuine
       }

   def print_result(self, result, is_genuine): pass