from utils.preprocessing import normalize_x_y_pressure
from .signature_verifier import SignatureVerifier
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, combine_pvalues
from sklearn.preprocessing import StandardScaler

class KolmogorovSmirnovSignatureVerifier(SignatureVerifier):

   def __init__(self, genuine_signature: pd.DataFrame, significance_level=0.01):
       self.genuine_features = self.preprocess_signature(genuine_signature)
       self.significance_level = significance_level
       self.scaler = StandardScaler()

   def get_col(self, df, names):
       """Умный поиск колонки: ищет 'X', ' X', 'X '..."""
       if isinstance(names, str):
           names = [names]
       
       for name in names:
           candidates = [name, ' ' + name, name + ' ']
           for cand in candidates:
               if cand in df.columns:
                   return np.array(df[cand])
       
       raise KeyError(f"Не найдены столбцы {names}. Доступно: {list(df.columns)}")

   def preprocess_signature(self, signature: pd.DataFrame, n_points=100) -> pd.DataFrame:
       x_data = self.get_col(signature, 'X')
       y_data = self.get_col(signature, 'Y')
       
       try:
           p_data = self.get_col(signature, ['Pressure', 'p', 'Pres'])
       except KeyError:
           p_data = np.zeros_like(x_data)

       (x, y, p) = normalize_x_y_pressure(x_data, y_data, p_data)

       new_sig = pd.DataFrame({
           'X': x,
           'Y': y,
           'Pressure': p
       })
       
       for col in ['Btn', 'Curvature', 'Speed', 'Acceleration', 'PressureVelocity']:
           try:
               new_sig[col] = self.get_col(signature, col)
           except KeyError:
               pass
               
       return new_sig

   def verify(self, test_signature: pd.DataFrame):
       if self.genuine_features is None:
           raise ValueError('Отсутствуют данные о настоящей подписи')
      
       test_features = self.preprocess_signature(test_signature)
      
       features = ['X', 'Y', 'Btn', 'Curvature', 'Speed', 'Acceleration', 'PressureVelocity']
       pvalues = []
       weights = [11, 9, 11, 2, 1, 2, 3]
       
       for feature in features:
           if feature in test_features.columns and feature in self.genuine_features.columns:
               ks_stat, p_value = ks_2samp(self.genuine_features[feature], test_features[feature])
               pvalues.append(p_value)
           else:
               pvalues.append(0.5)

       stat, combined_p_value = combine_pvalues(pvalues, weights=weights[:len(pvalues)], method='stouffer')
      
       return {
           'combined_p': combined_p_value,
           'is_genuine': combined_p_value > self.significance_level
       }
   
   def print_result(self, result, is_genuine): pass