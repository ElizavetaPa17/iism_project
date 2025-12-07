from .signature_verifier import SignatureVerifier
import colorama
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.preprocessing import normalize_x_y_pressure, resample_signature
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV
import logging

'''
Вычисление различных статистик и сравнение их при помощи такой метрики, как расстояние Махалонобиса
'''
class StatisticsSignatureVerifier(SignatureVerifier):
    
    def __init__(self, prior_genuine=0.5):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.robust_cov = MinCovDet(support_fraction=1)
        
        self.kde_genuine = None
        self.kde_forged = None
        self.logistic_calibrator = None
        self.isotonic_calibrator = None
        self.prior_genuine = prior_genuine # априорная вероятность того, что подпись подлинна

    def preprocess_signature(self, signature: pd.DataFrame, n_points=100) -> np.array:
        '''
        Предварительная обработка данных подписи (ресэмплинг и нормализация)
        для устранения временных вариаций и приведения данных к единому масштабу и форме.

            Параметры:
                sig - набор данных об исходной подписи
                n_points - количество точек, которые должны быть в результирующей выборке
            Результат:
                Массив, содержащий интерполированные и нормализованные значения координат X, Y и Pressure в диапазоне от 0 до 1,
                а также число сегментов в подписи
        '''
        sig_resampled = resample_signature(signature, n_points)

        x, y, p = sig_resampled[:, 0], sig_resampled[:, 1], sig_resampled[:, 2]
        (x_norm, y_norm, p_norm) = normalize_x_y_pressure(sig_resampled[:, 0], sig_resampled[:, 1], sig_resampled[:, 2])

        return np.column_stack((x_norm, y_norm, p_norm, x, y, p))
    
    def extract_features(self, signature: np.ndarray) -> np.ndarray:
        '''
        Вычисление статистических характеристик из подготовленных данных подписи для проведения дальнейшего анализа

            Параметры:
                signature - данные одной подписи, прошедшие ресэмплинг и нормализацию
            Результат:
                Массив, содержащий вычисленные характеристики для исходной подписи
        '''
        x, y, p = signature[:, 0], signature[:, 1], signature[:, 2]
        x_before, y_before, p_before = signature[:, 3], signature[:, 4], signature[:, 5]

        dx, dy = np.diff(x), np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2)
        acceleration = np.diff(velocity)

        if np.all(velocity == velocity[0]): velocity += 1e-9
        if np.all(acceleration == acceleration[0]): acceleration += 1e-9

        # Общая длина подписи
        length = np.sum(velocity)

        # Отношение общей длины подписи к евклидову расстоянию между начальной и конечной точками. 
        denominator = np.sqrt(
            np.power(np.max(x_before) - np.min(x_before), 2)
            + np.power(np.max(y_before) - np.min(y_before), 2)
        )

        points = np.column_stack((x_before, y_before))
        unique_points = np.unique(points, axis=0)
        hull_area = 0
        if len(unique_points) >= 3:
            hull = ConvexHull(unique_points)
            hull_area = hull.area

        features = [
            # пространственные
            np.mean(x_before),
            np.mean(y_before),
            (np.max(x) - np.min(x)) / (np.max(y) - np.min(y)) if (np.max(y) - np.min(y)) > 1e-9 else 0,
            hull_area,

            # динамические
            np.std(velocity), 
            length,
            stats.skew(velocity),
            stats.skew(acceleration),
            
            # давление
            np.max(p_before) - np.min(p_before),        
            stats.skew(p),
            
            # число изменений направления 
            len(np.where(np.diff(np.sign(acceleration)))[0]),

            # число пауз
            len(velocity[velocity < 0.1 * np.max(velocity)]) if np.max(velocity) > 0 else 0
        ]

        return np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)

    def load(self, genuine_signatures):
        '''
        Выполняет построение модели на основании подлинных подписей.
        '''
        self.N = math.ceil(np.min([len(sig) for sig in genuine_signatures if len(sig) > 0]))
        logging.debug(f'N: {self.N}')
        processed = [self.preprocess_signature(sig, self.N) for sig in genuine_signatures]

        self.features = np.array([self.extract_features(sig) for sig in processed])
        self.features_number = self.features.shape[1]

        transformed_features = self.scaler.fit_transform(self.features)
        self.robust_cov.fit(transformed_features)

        self.genuine_distances = self._calculate_distances(self.features)
        self.sorted_genuine_distances = sorted(self.genuine_distances)
        
        self.threshold = np.percentile(self.genuine_distances, 95)

    def load_forgeries_and_calibrate(self, forged_signatures):
        '''
        Вычисляет расстояния Махаланобиса для набора поддельных подписей для тренировки моделей.
        Метод должен быть вызван после load().
        '''
        if self.features is None:
            raise RuntimeError("Перед вызовом load_forgeries_and_calibrate() необходимо вызвать load() для набора подлинных подписей.")

        processed_forgeries = [self.preprocess_signature(sig, self.N) for sig in forged_signatures]
        forgery_features = np.array([self.extract_features(sig) for sig in processed_forgeries])
        self.forged_distances = self._calculate_distances(forgery_features)

        # Обучение KDE моделей
        genuine_dist_reshaped = self.genuine_distances.reshape(-1, 1)
        forged_dist_reshaped = self.forged_distances.reshape(-1, 1)

        # Перекрёстная проверка для нахождения оптимального значения bandwidth для KDE
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.logspace(-1, 1, 20)}, cv=3)
        
        grid.fit(genuine_dist_reshaped)
        self.kde_genuine = grid.best_estimator_
        logging.info(f"KDE Genuine bandwidth: {self.kde_genuine.bandwidth_}")

        grid.fit(forged_dist_reshaped)
        self.kde_forged = grid.best_estimator_
        logging.info(f"KDE Forged bandwidth: {self.kde_forged.bandwidth_}")

        # Обучение калибраторов.
        # Данные для обучения: X = distances, y = labels (1 - настоящие, 0 - поддельные)
        X_calib = np.concatenate((self.genuine_distances, self.forged_distances)).reshape(-1, 1)
        y_calib = np.concatenate((np.ones_like(self.genuine_distances), np.zeros_like(self.forged_distances)))

        self.logistic_calibrator = LogisticRegression()
        self.logistic_calibrator.fit(X_calib, y_calib)

        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(X_calib, y_calib)


    def _calculate_distances(self, features: np.ndarray) -> np.ndarray:
        '''
        Вычисление расстояний Махалонобиса - метрической меры,
        которая учитывает как расстояние точки от 'центра' данных (средних значений), 
        так и форму распределения (ковариационную матрицу)
        '''
        x_scaled = self.scaler.transform(features)
        return self.robust_cov.mahalanobis(x_scaled)
    
    def verify(self, test_signature: pd.DataFrame) -> dict:
        '''
        Тестирование подписи на подлинность. 

            Параметры:
                test_signature - необработанный набор данных подписи
            Результат:
                Словарь, содержащий расстояние Махаланобиса, пороговое значение и результат сравнения на подлинность
        '''
        test_processed = self.preprocess_signature(test_signature, self.N)
        test_features = self.extract_features(test_processed)
        
        distance = self._calculate_distances(np.array([test_features]))[0]

        # Вычисление эмпирических вероятностей P(D <= d | Genuine)
        p_ecdf = stats.percentileofscore(self.sorted_genuine_distances, distance, kind='weak') / 100.0
        
        prob_bayesian_kde = None
        prob_logistic = None
        prob_isotonic = None

        if self.kde_genuine and self.kde_forged:
            log_likelihood_genuine = self.kde_genuine.score_samples(np.array([[distance]]))[0]
            log_likelihood_forged = self.kde_forged.score_samples(np.array([[distance]]))[0]
            
            likelihood_genuine = np.exp(log_likelihood_genuine)
            likelihood_forged = np.exp(log_likelihood_forged)
            
            prior_genuine = self.prior_genuine
            prior_forged = 1 - prior_genuine
            
            evidence = (likelihood_genuine * prior_genuine) + (likelihood_forged * prior_forged)
            
            if evidence > 1e-9:
                # Теорема Байеса для нахождения P(Genuine | D = d) = P(D | Genuine) * P(Genuine) / P(D)
                prob_bayesian_kde = (likelihood_genuine * prior_genuine) / evidence
            else:
                prob_bayesian_kde = prior_genuine

        if self.logistic_calibrator:
            # predict_proba возвращает [[P(0), P(1)]]
            prob_logistic = self.logistic_calibrator.predict_proba(np.array([[distance]]))[0][1]

        if self.isotonic_calibrator:
            prob_isotonic = self.isotonic_calibrator.predict(np.array([[distance]]))[0]

        return {
            'distance': distance,
            'is_genuine': distance <= self.threshold,
            'p_ecdf': p_ecdf, # P(Dist <= d | Genuine)
            'threshold': self.threshold,
            'prob_bayesian_kde': prob_bayesian_kde, # P(Genuine | Dist = d)
            'prob_logistic': prob_logistic,         # P(Genuine | Dist = d)
            'prob_isotonic': prob_isotonic          # P(Genuine | Dist = d)
        }
    
    def plot_distributions(self):
        """
        Построение графиков распределения расстояний настоящих и поддельных подписей.
        """
        if self.forged_distances is None:
            logging.debug("Данные поддельных подписей не загружены, выполняем построение только распределения расстояний настоящих подписей.");
            plt.hist(self.genuine_distances, bins=20, density=True, alpha=0.6, color='green', label='Genuine расстояния')
            plt.axvline(self.threshold, color='red', linestyle='--', label=f'95% Threshold ({self.threshold:.2f})')
            plt.title('Гистограмма распределения расстояния Махаланобиса')
            plt.xlabel('Расстояние Махаланобиса')
            plt.ylabel('Частота')
            plt.legend()
            plt.show()
            return

        plt.figure(figsize=(12, 8))
        
        plt.hist(self.genuine_distances, bins=20, density=True, alpha=0.6, color='green', label='Genuine Расстояния')
        plt.hist(self.forged_distances, bins=20, density=True, alpha=0.6, color='red', label='Forged Расстояния')
        
        # KDE-кривые
        x_plot = np.linspace(0, max(max(self.genuine_distances), max(self.forged_distances)), 1000)[:, np.newaxis]
        if self.kde_genuine:
            log_dens_g = self.kde_genuine.score_samples(x_plot)
            plt.plot(x_plot[:, 0], np.exp(log_dens_g), color='darkgreen', lw=2, linestyle='-', label='Genuine KDE')
        if self.kde_forged:
            log_dens_f = self.kde_forged.score_samples(x_plot)
            plt.plot(x_plot[:, 0], np.exp(log_dens_f), color='darkred', lw=2, linestyle='-', label='Forged KDE')
            
        plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
        plt.title('Гистограмма распределения расстояния Махаланобиса и KDE кривые')
        plt.xlabel('Расстояние Махаланобиса')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_result(self, result: dict, is_genuine: bool):
        is_genuine_by_threshold = result['is_genuine']
        color = colorama.Fore.GREEN if is_genuine == is_genuine_by_threshold else colorama.Fore.RED

        print(f"Расстояние: {result['distance']:.3f}, вердикт: " + color + f"{is_genuine_by_threshold}" + colorama.Fore.RESET)
        print(f"p_ecdf: {result['p_ecdf']:.3f}, p_bayesian_kde: {result['prob_bayesian_kde']:.3f}, p_logistic: {result['prob_logistic']:.3f}, p_isotonic: {result['prob_isotonic']:.3f}")
        print("-" * 30)