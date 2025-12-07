import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_signature

from task1.ks_verifier import KolmogorovSmirnovSignatureVerifier as OldVerifier
from task1.ks_verifier2 import KolmogorovSmirnovSignatureVerifier as NewVerifier

FOLDER_PATH = '/home/panteriza/Downloads/Telegram Desktop/SignEEGv1.0/000000000200894/Genuine'

def get_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')])

def main():
    files = get_files(FOLDER_PATH)
    if len(files) < 2: return

    test_files = files[:5]
    
    genuine_sig = load_signature(test_files[0])
    old_verifier = OldVerifier(genuine_sig)
    new_verifier = NewVerifier(genuine_sig)
    
    filenames = []
    old_values = [] 
    new_values = [] 

    print("Расчет метрик...")
    for f_path in test_files[1:]: 
        name = os.path.basename(f_path)
        sig = load_signature(f_path)
        
        res_old = old_verifier.verify(sig)
        p_val = res_old['combined_p']
        if p_val == 0: p_val = 1e-50 
        old_values.append(p_val)
        
        res_new = new_verifier.verify(sig)
        new_values.append(res_new['probability_forgery2'] * 100)
        
        filenames.append(f"...{name[-10:]}") 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_old = ['gray'] * len(old_values)
    bars1 = ax1.bar(filenames, old_values, color=colors_old, alpha=0.6)
    
    ax1.set_yscale('log')
    ax1.set_title('Старый метод (KS P-value)\nШкала: Логарифмическая', fontsize=12, fontweight='bold')
    ax1.set_ylabel('P-value (Вероятность совпадения)')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'$10^{{{int(np.log10(height))}}}$',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', color='red', fontsize=10)

    colors_new = ['green'] * len(new_values)
    bars2 = ax2.bar(filenames, new_values, color=colors_new, alpha=0.8)
    
    ax2.set_ylim(0, 110)
    ax2.set_title('Новый метод (Модифицированный)\nШкала: Линейная (Проценты)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Вероятность подлинности (%)')
    ax2.axhline(y=50, color='red', linestyle='--', label='Порог')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle("ЭВОЛЮЦИЯ МЕТОДА: От математического шума к реальному результату", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()