import sys
import os
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

# СТРОКА РАСКОММЕНТИРОВАНА ТОЛЬКО ТОГДА, КОГДА ЗАПУСК ПРОИСХОДИТ НА ЛИНУКС
os.environ["QT_QPA_PLATFORM"] = "xcb"

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
