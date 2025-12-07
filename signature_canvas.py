import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class SignatureCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super(SignatureCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.clear_plot()

    def clear_plot(self):
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.text(0.5, 0.5, "Выберите файл\nдля просмотра",
                       horizontalalignment='center', verticalalignment='center',
                       transform=self.axes.transAxes, color='gray', fontsize=12)
        self.draw()

    def plot_signature(self, df):
        self.axes.clear()
        self.axes.set_axis_off()

        if df is None or df.empty:
            self.axes.text(0.5, 0.5, "Пустой файл или\nошибка чтения данных",
                           ha='center', va='center', transform=self.axes.transAxes, color='red')
            self.draw()
            return

        try:
            if 'X' not in df.columns or 'Y' not in df.columns:
                raise ValueError(f"Нет колонок X/Y. Есть: {list(df.columns)}")

            x = df['X'].values
            y = df['Y'].values

            if 'Pressure' in df.columns:
                p = df['Pressure'].values
                if np.ptp(p) == 0:
                    p = np.ones_like(p) * 20
                else:
                    p = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-9)
                    p = p * 40 + 5
            else:
                p = 10

            self.axes.plot(x, y, color='blue', linewidth=1, alpha=0.3)
            self.axes.scatter(x, y, s=p, c='darkblue', alpha=0.6, edgecolors='none')

            self.axes.invert_yaxis()
            self.axes.set_aspect('equal', 'datalim')

        except Exception as e:
            self.axes.clear()
            self.axes.set_axis_off()
            self.axes.text(0.5, 0.5, f"Ошибка отрисовки:\n{str(e)}",
                           ha='center', va='center', transform=self.axes.transAxes, color='red')
            print(f"Plot Error: {e}") 

        self.draw()
