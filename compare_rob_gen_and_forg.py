import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_signature

from task1.robust import RobustKolmogorovSmirnovVerifier

GENUINE_FOLDER = '/home/panteriza/Downloads/Telegram Desktop/SignEEGv1.0/000000814510023/Genuine'
FORGED_FOLDER = '/home/panteriza/Downloads/Telegram Desktop/SignEEGv1.0/000000814510023/Forged'

def get_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')])

def main():
    gen_files = get_files(GENUINE_FOLDER)
    forg_files = get_files(FORGED_FOLDER)
    
    if len(gen_files) < 5:
        print("Ошибка: Для хорошего теста нужно минимум 5 файлов Genuine (3 на обучение, 2 на тест).")
        return

    TRAIN_COUNT = 3
    print(f"Формирование профиля по {TRAIN_COUNT} эталонам...")
    
    training_sigs = []
    for f in gen_files[:TRAIN_COUNT]:
        training_sigs.append(load_signature(f))
        
    verifier = RobustKolmogorovSmirnovVerifier(training_sigs)
    
    labels = []
    scores = []
    colors = []

    print("\nПроверка настоящих (Test Genuine)...")
    for f_path in gen_files[TRAIN_COUNT:TRAIN_COUNT+4]: 
        sig = load_signature(f_path)
        res = verifier.verify(sig)
        
        votes = res['pass_votes']
        score = (votes / 8) * 100 
        
        labels.append("Genuine")
        scores.append(score)
        colors.append('green')
        print(f"  {os.path.basename(f_path)} -> Голосов: {votes}/8 ({score:.1f}%)")

    print("\nПроверка поддельных (Forged)...")
    for f_path in forg_files[:4]:
        sig = load_signature(f_path)
        res = verifier.verify(sig)
        
        votes = res['pass_votes']
        score = (votes / 8) * 100
        
        labels.append("Forged")
        scores.append(score)
        colors.append('red')
        print(f"  {os.path.basename(f_path)} -> Голосов: {votes}/8 ({score:.1f}%)")

    x = np.arange(len(scores))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, scores, color=colors, alpha=0.8)

    ax.set_ylabel('Уверенность (Процент голосов)')
    ax.set_title('Робастный метод (Голосование)\n(Зеленые - свои, Красные - чужие)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    
    ax.axhline(y=50, color='black', linestyle='--', label='Порог (4 голоса)')
    ax.legend()

    for bar in bars:
        height = bar.get_height()
        votes = int(height / 100 * 8)
        ax.annotate(f'{votes}/8',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()