import pandas as pd
import numpy as np
import logging

from utils.preprocessing import load_signature

from task1.ks_verifier import KolmogorovSmirnovSignatureVerifier as SimpleKSVerifier
from task1.statistics_verifier import StatisticsSignatureVerifier
from task1.robust import RobustKolmogorovSmirnovVerifier
from task1.ks_verifier2 import KolmogorovSmirnovSignatureVerifier as BayesianKSVerifier

class VerificationManager:
    METHODS = [
        "KS Test (Simple)",
        "KS Test (Bayesian)",
        "Statistical (Mahalanobis)",
        "Robust (Voting)"
    ]

    def __init__(self):
        self.genuine_signatures = []
        self.current_verifier = None
        self.current_method_name = ""

    def load_genuine_signatures(self, file_paths):
        self.genuine_signatures = []
        loaded_count = 0
        for path in file_paths:
            try:
                df = load_signature(path)
                if not df.empty:
                    self.genuine_signatures.append(df)
                    loaded_count += 1
            except Exception as e:
                print(f"Ошибка загрузки {path}: {e}")
        return loaded_count

    def train_model(self, method_name):
        if not self.genuine_signatures:
            raise ValueError("Сначала загрузите эталонные подписи!")

        self.current_method_name = method_name

        if method_name == "KS Test (Simple)":
            self.current_verifier = SimpleKSVerifier(self.genuine_signatures[0])

        elif method_name == "KS Test (Bayesian)":
            self.current_verifier = BayesianKSVerifier(self.genuine_signatures[0])

        elif method_name == "Statistical (Mahalanobis)":
            self.current_verifier = StatisticsSignatureVerifier()
            self.current_verifier.load(self.genuine_signatures)

        elif method_name == "Robust (Voting)":
            self.current_verifier = RobustKolmogorovSmirnovVerifier(self.genuine_signatures)
            self.current_verifier.pass_threshold = 4

    def verify_signature(self, file_path):
        if not self.current_verifier:
            raise ValueError("Модель не инициализирована")

        try:
            test_signature = load_signature(file_path)
            if test_signature.empty: return {"error": "Пустой файл"}

            result_raw = self.current_verifier.verify(test_signature)
            return self._standardize_result(result_raw)

        except Exception as e:
            return {"error": str(e), "is_genuine": False, "confidence": 0.0, "details": str(e)}

    def _standardize_result(self, raw_result):
        response = {
            "is_genuine": False,
            "confidence": 0.0,
            "details": ""
        }

        # 1. SIMPLE KS
        if self.current_method_name == "KS Test (Simple)":
            response["is_genuine"] = raw_result['is_genuine']
            response["confidence"] = min(raw_result['combined_p'] * 100, 100)
            response["details"] = f"P-value: {raw_result['combined_p']:.4f}"

        # 2. BAYESIAN KS (Новый)
        elif self.current_method_name == "KS Test (Bayesian)":
            response["is_genuine"] = raw_result['is_genuine']
            prob = raw_result.get('probability_forgery2', 0)
            response["confidence"] = prob * 100
            response["details"] = f"Bayes Prob: {prob:.4f}"

        # 3. STATISTICAL
        elif self.current_method_name == "Statistical (Mahalanobis)":
            response["is_genuine"] = raw_result['is_genuine']
            if raw_result.get('prob_bayesian_kde') is not None:
                response["confidence"] = raw_result['prob_bayesian_kde'] * 100
            else:
                response["confidence"] = raw_result.get('p_ecdf', 0) * 100

            dist = raw_result['distance']
            thresh = raw_result.get('threshold', 0)
            response["details"] = f"Dist: {dist:.2f} (Tr: {thresh:.2f})"

        # 4. ROBUST
        elif self.current_method_name == "Robust (Voting)":
            response["is_genuine"] = raw_result['is_genuine']
            votes = raw_result['pass_votes']
            total = 8
            response["confidence"] = (votes / total) * 100
            response["details"] = f"Votes: {votes}/{total}"

        return response
