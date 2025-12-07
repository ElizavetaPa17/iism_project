from utils.preprocessing import load_from_folder
from utils.files import forged_exists, construct_signature_path
from .ks_verifier import KolmogorovSmirnovSignatureVerifier
from .statistics_verifier import StatisticsSignatureVerifier
import argparse
import colorama
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def color_print(text, color):
    '''
    Вывод строки в терминал с определенным цветом текста
    
        Параметры:
            text - строка для вывода
            color - цвет текста
    '''
    print(color + text + colorama.Fore.RESET)


def main():
    np.random.default_rng(seed=42)

    parser = argparse.ArgumentParser("main")
    parser.add_argument("method", help="method used for signature verification ('stats'/'ks')", type=str)
    parser.add_argument("-p", "--plot", help="turn on plotting histograms", action="store_true")
    args = parser.parse_args()

    it = 1
    fp_s, fn_s = [], []
    accuracy_s, spec_s, sens_s = [], [], []

    try:
        all_genuine_folders = []
        all_forged_folders = []

        for person in forged_exists:
            all_genuine_folders.append([construct_signature_path(person)])
            all_forged_folders.append(construct_signature_path(person, is_genuine=False))

        print(all_genuine_folders, all_forged_folders)

        for i, genuine_folders in enumerate(all_genuine_folders):
            color_print('-'*25 + str(it) + '-' * 25, colorama.Fore.BLUE)
            it += 1

            forged_folder = all_forged_folders[i]

            additional_features = {'Speed', 'Acceleration', 'Curvature', 'PressureVelocity'}
                                        
            genuine_signatures = []
            for genuine_folder in genuine_folders:
                genuine_signatures.extend(load_from_folder(genuine_folder, additional_features))

            forged_signatures = load_from_folder(forged_folder, additional_features)

            verifier = None
            if args.method == 'stats':
                verifier = StatisticsSignatureVerifier()
                verifier.load(genuine_signatures)
                color_print(f'Threshold: {verifier.threshold:.3f}', colorama.Fore.CYAN)
                if args.plot:
                    verifier.plot_distribution()
            else:
                verifier = KolmogorovSmirnovSignatureVerifier(genuine_signatures[0])
                color_print(f'Уровень значимости: {verifier.significance_level:.2f}', colorama.Fore.CYAN)

            # tn - настоящая определена как настоящая
            # fp - настоящая определена как поддельная
            # tp - поддельная определена как поддельная
            # fn - поддельная определена как настоящая
            (tp, fp, tn, fn) = (0, 0, 0, 0)

            color_print('Тестирование настоящих подписей:', colorama.Fore.BLUE)
            color_print(', '.join(genuine_folders).replace('/home/polina', '~'), colorama.Fore.BLUE)

            for genuine_sig in genuine_signatures:
                result = verifier.verify(genuine_sig)
                if result['is_genuine']:
                    tn += 1
                else:
                    fp += 1
                verifier.print_result(result, True)

            color_print('Тестирование исследуемой подписи:', colorama.Fore.BLUE)
            color_print(forged_folder.replace('/home/polina', '~'), colorama.Fore.BLUE)
            for forged_sig in forged_signatures:
                result = verifier.verify(forged_sig)
                if result['is_genuine']:
                    fn += 1
                else:
                    tp += 1
                verifier.print_result(result, False)
                                    
            total_count = len(genuine_signatures) + len(forged_signatures)

            color_print('Статистика по настоящим подписям:', colorama.Fore.BLUE)
            fp_s.append(fp/len(genuine_signatures)*100)
            fn_s.append(fn/len(forged_signatures)*100)
            print(f'Процент настоящих, определенных как поддельные (FP): {fp_s[-1]:.2f}%')
            print(f'Процент поддельных, определенных как настоящие (FN): {fn_s[-1]:.2f}%')

            # точность - процент правильно определенных подписей
            accuracy = (tp + tn) / total_count
            accuracy_s.append(accuracy*100)
            print(f'Точность: {accuracy * 100:.2f}%')

            # чувствительность - корректно определенные поддельные подписи
            sensitivity = tp / (tp + fn)
            sens_s.append(sensitivity*100)
            print(f'Чувствительность: {sensitivity * 100:.2f}%')

            # корректно определенные настоящие подписи
            specificity = tn / (tn + fp)
            spec_s.append(specificity*100)
            print(f'Специфичность: {specificity * 100:.2f}%')

    except Exception as e:
        print(f"Произошла ошибка: {e}")

    color_print('Итоговая статистика:', colorama.Fore.BLUE)
    print(f'FP: mean={np.mean(fp_s):.2f}%, min={np.min(fp_s):.2}%, max={np.max(fp_s):.2f}%')
    print(f'FN: mean={np.mean(fn_s):.2f}%, min={np.min(fn_s):.2}%, max={np.max(fn_s):.2f}%')
    print(f'Точность: mean={np.mean(accuracy_s):.2f}%, min={np.min(accuracy_s):.2f}%, max={np.max(accuracy_s):.2f}%')
    print(f'Чувствительность: mean={np.mean(sens_s):.2f}%, min={np.min(sens_s):.2f}%, max={np.max(sens_s):.2f}%')
    print(f'Специфичность: mean={np.mean(spec_s):.2f}%, min={np.min(spec_s):.2f}%, max={np.max(spec_s):.2f}%')

    return


if __name__ == "__main__":
    main()