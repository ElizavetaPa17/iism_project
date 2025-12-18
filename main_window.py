import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget,
                             QComboBox, QTableWidget, QTableWidgetItem,
                             QSplitter, QHeaderView, QMessageBox, QGroupBox,
                             QAbstractItemView, QScrollArea, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from signature_canvas import SignatureCanvas
from verification_manager import VerificationManager
from utils.preprocessing import load_signature

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Signature Verification System")
        self.resize(1200, 800)

        self.manager = VerificationManager()
        self.genuine_paths = []
        self.test_paths = []

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(320) 

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15) 

        gb_genuine = QGroupBox("1. Эталоны (Genuine)")
        gb_genuine.setStyleSheet("QGroupBox { font-weight: bold; }")
        gb_gen_layout = QVBoxLayout()

        self.btn_load_genuine = QPushButton("📂 Загрузить эталоны")
        self.btn_load_genuine.setFixedHeight(40) 
        self.btn_load_genuine.clicked.connect(self.load_genuine_files)

        self.list_genuine = QListWidget()
        self.list_genuine.setMaximumHeight(150) 
        self.list_genuine.itemClicked.connect(lambda item: self.display_signature(item, is_genuine=True))

        gb_gen_layout.addWidget(self.btn_load_genuine)
        gb_gen_layout.addWidget(self.list_genuine)
        gb_genuine.setLayout(gb_gen_layout)

        gb_forged = QGroupBox("2. Подделки (Forged)")
        gb_forged.setStyleSheet("QGroupBox { font-weight: bold; }")
        gb_forged_layout = QVBoxLayout()

        self.btn_load_forged = QPushButton("📂 Загрузить подделки")
        self.btn_load_forged.setFixedHeight(40) 
        self.btn_load_forged.clicked.connect(self.load_forged_files)

        self.list_forged = QListWidget()
        self.list_forged.setMaximumHeight(150) 
        self.list_forged.itemClicked.connect(lambda item: self.display_signature(item, is_genuine=False))

        gb_forged_layout.addWidget(self.btn_load_forged)
        gb_forged_layout.addWidget(self.list_forged)
        gb_forged.setLayout(gb_forged_layout)

        gb_method = QGroupBox("3. Метод проверки")
        gb_method.setStyleSheet("QGroupBox { font-weight: bold; }")
        gb_meth_layout = QVBoxLayout()

        self.combo_method = QComboBox()
        self.combo_method.addItems(self.manager.METHODS)
        self.combo_method.setFixedHeight(30)

        self.btn_train = QPushButton("⚙️ Инициализировать")
        self.btn_train.setFixedHeight(40)
        self.btn_train.setStyleSheet("background-color: #d4e3fc; font-weight: bold; border-radius: 4px; border: 1px solid #a0a0a0;")
        self.btn_train.clicked.connect(self.init_model)

        gb_meth_layout.addWidget(QLabel("Алгоритм:"))
        gb_meth_layout.addWidget(self.combo_method)
        gb_meth_layout.addWidget(self.btn_train)
        gb_method.setLayout(gb_meth_layout)

        gb_test = QGroupBox("4. Проверка (Test)")
        gb_test.setStyleSheet("QGroupBox { font-weight: bold; }")
        gb_test_layout = QVBoxLayout()

        self.btn_load_test = QPushButton("📂 Загрузить файлы")
        self.btn_load_test.setFixedHeight(40)
        self.btn_load_test.clicked.connect(self.load_test_files)

        self.list_test = QListWidget()
        self.list_test.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_test.itemClicked.connect(lambda item: self.display_signature(item, is_genuine=False))

        self.btn_verify = QPushButton("✅ ПРОВЕРИТЬ")
        self.btn_verify.setFixedHeight(60)
        self.btn_verify.setCursor(Qt.PointingHandCursor)
        self.btn_verify.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.btn_verify.clicked.connect(self.run_verification)

        gb_test_layout.addWidget(self.btn_load_test)
        gb_test_layout.addWidget(self.list_test)
        gb_test_layout.addWidget(self.btn_verify)
        gb_test.setLayout(gb_test_layout)

        left_layout.addWidget(gb_genuine)
        left_layout.addWidget(gb_forged)
        left_layout.addWidget(gb_method)
        left_layout.addWidget(gb_test)
        left_layout.addStretch() 

        left_scroll.setWidget(left_widget)
        self.main_splitter.addWidget(left_scroll)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_splitter = QSplitter(Qt.Vertical)

        self.canvas = SignatureCanvas(self, width=5, height=4, dpi=100)
        right_splitter.addWidget(self.canvas)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(5)
        self.table_results.setHorizontalHeaderLabels(["Файл", "Метод", "Вердикт", "Вероятность", "Детали"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_results.setAlternatingRowColors(True) 
        self.table_results.itemClicked.connect(self.on_table_click)

        right_splitter.addWidget(self.table_results)
        right_splitter.setSizes([600, 300]) 

        self.main_splitter.addWidget(right_splitter)

        self.main_splitter.setSizes([350, 850])

        self.statusBar().showMessage("Готов к работе")

    def load_genuine_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите эталонные подписи", "", "CSV Files (*.csv);;All Files (*)")
        if files:
            self.genuine_paths = files
            self.list_genuine.clear()
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.list_genuine.addItem(item)

            count = self.manager.load_genuine_signatures(self.genuine_paths)
            self.statusBar().showMessage(f"Загружено {count} эталонных подписей.")
            QMessageBox.information(self, "Успех", f"Загружено {count} эталонных подписей.")

    def load_forged_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите поддельные подписи", "", "CSV Files (*.csv);;All Files (*)")
        if files:
            self.forged_paths = files
            self.list_forged.clear()
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.list_forged.addItem(item)

            count = self.manager.load_forged_signatures(self.forged_paths)
            self.statusBar().showMessage(f"Загружено {count} поддельных подписей.")
            QMessageBox.information(self, "Успех", f"Загружено {count} поддельных подписей.")

    def load_test_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы для проверки", "", "CSV Files (*.csv);;All Files (*)")
        if files:
            self.test_paths = files
            self.list_test.clear()
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.list_test.addItem(item)
            self.statusBar().showMessage(f"Добавлено {len(files)} файлов для проверки.")

    def init_model(self):
        method = self.combo_method.currentText()
        try:
            if not self.genuine_paths:
                QMessageBox.warning(self, "Ошибка", "Нет эталонных файлов!")
                return

            self.manager.load_genuine_signatures(self.genuine_paths)
            self.manager.train_model(method)

            self.statusBar().showMessage(f"Модель '{method}' успешно инициализирована.")
            QMessageBox.information(self, "Готово", f"Метод '{method}' настроен и готов к проверке.")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_verification(self):
        if not self.manager.current_verifier:
            QMessageBox.warning(self, "Внимание", "Сначала нажмите кнопку 'Инициализировать'!")
            return

        if self.list_test.count() == 0:
             QMessageBox.warning(self, "Внимание", "Нет файлов для проверки!")
             return

        self.table_results.setRowCount(0)

        for i in range(self.list_test.count()):
            item = self.list_test.item(i)
            file_path = item.data(Qt.UserRole)
            filename = item.text()

            res = self.manager.verify_signature(file_path)

            row_idx = self.table_results.rowCount()
            self.table_results.insertRow(row_idx)

            file_item = QTableWidgetItem(filename)
            file_item.setData(Qt.UserRole, file_path)
            self.table_results.setItem(row_idx, 0, file_item)

            self.table_results.setItem(row_idx, 1, QTableWidgetItem(self.combo_method.currentText()))

            verdict_text = "GENUINE" if res['is_genuine'] else "FORGED"
            verdict_item = QTableWidgetItem(verdict_text)
            verdict_item.setFont(QFont("Arial", 10, QFont.Bold))
            verdict_item.setTextAlignment(Qt.AlignCenter) # Выравнивание по центру
            if res['is_genuine']:
                verdict_item.setForeground(QColor("green"))
                verdict_item.setBackground(QColor(230, 255, 230))
            else:
                verdict_item.setForeground(QColor("red"))
                verdict_item.setBackground(QColor(255, 230, 230))
            self.table_results.setItem(row_idx, 2, verdict_item)

            conf_val = res['confidence']
            conf_item = QTableWidgetItem(f"{conf_val:.2f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.table_results.setItem(row_idx, 3, conf_item)

            labels = ["Файл", "Метод", "Вердикт", "Вероятность"]

            if res.get('confidence_log') is not None:
                labels.append("Вероятность (logistic)")
                conf_log_val = res['confidence_log']
                conf_log_item = QTableWidgetItem(f"{conf_log_val:.2f}%")
                conf_log_item.setTextAlignment(Qt.AlignCenter)
                self.table_results.setItem(row_idx, 4, conf_log_item)

            if res.get('confidence_iso') is not None:
                labels.append("Вероятность (isotonic)")
                conf_iso_val = res['confidence_iso']
                conf_iso_item = QTableWidgetItem(f"{conf_iso_val:.2f}%")
                conf_iso_item.setTextAlignment(Qt.AlignCenter)
                self.table_results.setItem(row_idx, 5, conf_iso_item)

            labels.append("Детали")
            self.table_results.setHorizontalHeaderLabels(labels)
            self.table_results.setColumnCount(len(labels))
            self.table_results.setItem(row_idx, 6, QTableWidgetItem(str(res['details'])))

        self.manager.plot()

        self.statusBar().showMessage("Проверка завершена.")

    def display_signature(self, item, is_genuine=True):
        file_path = item.data(Qt.UserRole)
        try:
            df = load_signature(file_path)
            self.canvas.plot_signature(df)
            sig_type = "Эталон" if is_genuine else "Тест"
            self.statusBar().showMessage(f"Просмотр: {os.path.basename(file_path)} ({sig_type})")
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка чтения файла: {e}")

    def on_table_click(self, item):
        row = item.row()
        file_item = self.table_results.item(row, 0)
        file_path = file_item.data(Qt.UserRole)

        try:
            df = load_signature(file_path)
            self.canvas.plot_signature(df)
            self.statusBar().showMessage(f"Просмотр результата: {os.path.basename(file_path)}")
        except Exception as e:
            print(e)

from PyQt5.QtWidgets import QListWidgetItem
