import os
import glob
import math
import pandas as pd
import numpy as np
import logging
from scipy import interpolate
from scipy.stats import ks_2samp, combine_pvalues
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

def calculate_metrics(df: pd.DataFrame, metrics: set) -> pd.DataFrame:
    df = df.copy()
    
    dx = np.diff(df['X'], prepend=df['X'].iloc[0])
    dy = np.diff(df['Y'], prepend=df['Y'].iloc[0])
    dt = np.diff(df['T'], prepend=df['T'].iloc[0])
    
    distance = np.sqrt(dx**2 + dy**2)

    speed = np.divide(distance, dt, out=np.zeros_like(distance, dtype=float), where=dt!=0)
    
    acceleration = np.gradient(speed)
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature_numerator = (dx * ddy - dy * ddx)
    curvature_denominator = (dx**2 + dy**2)**1.5
    curvature = np.divide(curvature_numerator, curvature_denominator, 
                          out=np.zeros_like(curvature_numerator, dtype=float), 
                          where=curvature_denominator!=0)

    dp = np.diff(df['Pressure'], prepend=df['Pressure'].iloc[0])

    if 'Speed' in metrics:
        df['Speed'] = speed
    if 'Curvature' in metrics:
        df['Curvature'] = curvature
    if 'Acceleration' in metrics:
        df['Acceleration'] = acceleration
    if 'PressureVelocity' in metrics:
        df['PressureVelocity'] = dp
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def get_file_list(folder_path: str) -> list:
    files = []
    for ext in ('*.csv', '*.txt'):
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return files

def load_signature(file_path: str, additional_metrics: set=None) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=r'\s*,\s*', engine='python', skiprows=1)
    except Exception as e:
        if "No columns to parse from file" in str(e): return pd.DataFrame()
        raise e

    required_columns =  {'Stroke', 'Btn', 'X', 'Y', 'T', 'Pressure'}
    if not required_columns.issubset(df.columns):
        print("Столбцы, которые увидел pandas:", df.columns.tolist())
        raise ValueError(f"В файле {file_path} отсутствуют необходимые столбцы: {required_columns - set(df.columns)}")

    if df['Stroke'].dtype == 'object':
        df['Stroke'] = [int(str(stroke).split()[0]) for stroke in df['Stroke']]

    return df

def load_from_folder(folder: str, additional_features: set) -> list[pd.DataFrame]:
    if not folder or not os.path.exists(folder): return []
    res = []
    for file in get_file_list(folder):
        try:
            res.append(load_signature(file, additional_features))
        except Exception as e:
            print(f"  -> Не удалось загрузить или обработать файл {os.path.basename(file)}: {e}")
    return [df for df in res if not df.empty]

def preprocess_pressure(pressure_series: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(np.array(pressure_series).reshape(-1, 1)).flatten()

def normalize_x_y_pressure(x, y, p) -> tuple:
    x_norm = MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten()
    y_norm = MinMaxScaler().fit_transform(np.array(y).reshape(-1, 1)).flatten()
    p_norm = preprocess_pressure(np.array(p))
    return (x_norm, y_norm, p_norm)

class RobustKolmogorovSmirnovVerifier:
    def __init__(self, genuine_signatures: list[pd.DataFrame], significance_level=0.05):
        """
        Инициализируется СПИСКОМ подлинных подписей для создания устойчивого профиля.
        """
        self.significance_level = significance_level
        self.pass_threshold = 0  # Порог голосов, который мы подберем
        
        self.feature_columns = ['X', 'Y', 'Btn', 'Curvature', 'Speed', 'Acceleration', 'PressureVelocity', 'Pressure']
        self.reference_profile = {}

        all_genuine_dfs = []
        for sig in genuine_signatures:
            metrics_to_calc = {'Speed', 'Acceleration', 'Curvature', 'PressureVelocity'}
            sig_with_metrics = calculate_metrics(sig, metrics_to_calc)
            all_genuine_dfs.append(self.preprocess_signature(sig_with_metrics))

        for feature in self.feature_columns:
            combined_series = pd.concat([df[feature] for df in all_genuine_dfs if feature in df.columns], ignore_index=True)
            if not combined_series.empty:
                self.reference_profile[feature] = combined_series

    def preprocess_signature(self, signature: pd.DataFrame) -> pd.DataFrame:
        sig_copy = signature.copy()
        (x, y, p) = normalize_x_y_pressure(
            np.array(sig_copy['X']), 
            np.array(sig_copy['Y']), 
            np.array(sig_copy['Pressure'])
        )
        sig_copy['X'] = x
        sig_copy['Y'] = y
        sig_copy['Pressure'] = p
        return sig_copy
    
    def verify(self, test_signature: pd.DataFrame):
        """
        Проверяет подпись, используя систему голосования.
        """
        metrics_to_calc = {'Speed', 'Acceleration', 'Curvature', 'PressureVelocity'}
        test_sig_with_metrics = calculate_metrics(test_signature, metrics_to_calc)
        test_features = self.preprocess_signature(test_sig_with_metrics)

        pass_votes = 0
        for feature in self.feature_columns:
            if feature in self.reference_profile and feature in test_features.columns:
                # Проводим KS-тест против объединенного профиля
                ks_stat, p_value = ks_2samp(self.reference_profile[feature], test_features[feature])
                
                # Если p-value достаточно большое, засчитываем голос
                if p_value > self.significance_level:
                    pass_votes += 1
        
        return {
            'pass_votes': pass_votes,
            'is_genuine': pass_votes >= self.pass_threshold
        }

    def find_best_threshold(self, genuine_signatures, forged_signatures):
        """
        Подбирает оптимальный порог голосов на основе данных.
        """
        votes = []
        labels = []

        for sig in genuine_signatures:
            votes.append(self.verify(sig)['pass_votes'])
            labels.append(1) # 1 = genuine

        for sig in forged_signatures:
            votes.append(self.verify(sig)['pass_votes'])
            labels.append(0) # 0 = forged
        
        best_accuracy = 0
        best_threshold = 0

        num_features = len(self.reference_profile)
        for t in range(1, num_features + 1):
            predictions = [1 if v >= t else 0 for v in votes]
            accuracy = accuracy_score(labels, predictions)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        
        self.pass_threshold = best_threshold
        return best_threshold, best_accuracy

def convert_p_to_probability_log(p_value, significance_level, steepness):
    log_p = -np.log(p_value + 1e-10) 
    x = significance_level + steepness * log_p
    prob_forgery = 1 / (1 + math.exp(-x))
    return 1 - prob_forgery

def p_to_posterior_prob(p_value, prior_prob_forgery):
    p_value = max(p_value, 1e-10)
    if p_value < 1/np.e:
        bayes_factor_forgery = -1 / (np.e * p_value * np.log(p_value))
    else:
        bayes_factor_forgery = 1.0
    prior_odds = prior_prob_forgery / (1 - prior_prob_forgery)
    posterior_odds = bayes_factor_forgery * prior_odds
    posterior_prob_forgery = posterior_odds / (1 + posterior_odds)
    return 1 - posterior_prob_forgery

if __name__ == "__main__":
    ROOT_DATA_DIR = '/home/panteriza/Downloads/Telegram Desktop/SignEEGv1.0' 
    
    FIXED_VOTE_THRESHOLD = 4
    
    all_accuracies = []
    all_specificities = []
    all_fp_rates = []

    user_folders = [f.path for f in os.scandir(ROOT_DATA_DIR) if f.is_dir()]
    
    for user_path in user_folders:
        user_id = os.path.basename(user_path)
        print(f"Пользователь {user_id}")
        
        genuine_folder = os.path.join(user_path, 'Genuine')
        forged_folder = os.path.join(user_path, 'Forged')

        genuine_signatures = load_from_folder(genuine_folder, set())
        forged_signatures = load_from_folder(forged_folder, set())

        if len(genuine_signatures) < 2 or not forged_signatures:
            continue

        verifier = RobustKolmogorovSmirnovVerifier(genuine_signatures)

        verifier.pass_threshold = FIXED_VOTE_THRESHOLD
        
        tn, fp, tp, fn = 0, 0, 0, 0
        
        for sig in genuine_signatures:
            if verifier.verify(sig)['is_genuine']: tn += 1
            else: fp += 1
        
        for sig in forged_signatures:
            if not verifier.verify(sig)['is_genuine']: tp += 1
            else: fn += 1

        total = tn + fp + tp + fn
        accuracy = (tn + tp) / total if total > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0

        all_accuracies.append(accuracy)
        all_specificities.append(specificity)
        all_fp_rates.append(fp_rate)

    print("\n" + "="*50)
    print(f"ИТОГОВАЯ СТАТИСТИКА (ГЛОБАЛЬНЫЙ ПОРОГ = {FIXED_VOTE_THRESHOLD})")
    print("="*50)
    if all_accuracies:
        print(f"Средняя Точность:      {np.mean(all_accuracies):.2%}")
        print(f"Средняя Специфичность: {np.mean(all_specificities):.2%}")
        print(f"Средний FP Rate:       {np.mean(all_fp_rates):.2%}")
    else:
        print("Не удалось обработать ни одного пользователя.")

