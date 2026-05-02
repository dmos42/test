import sys
import os
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import stats as scipy_stats

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QSplitter, QMenuBar, QMenu, QAction, QDialog, QDialogButtonBox, QStatusBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame, QToolBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QFont, QKeySequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats_engine.capability import CapabilityAnalysis
from stats_engine.normality import NormalityTests
from stats_engine.outliers import OutlierDetection


NUM_COLS = 26
NUM_ROWS = 500


def col_letter(n):
    result = ""
    while n >= 0:
        result = chr(65 + (n % 26)) + result
        n = n // 26 - 1
    return result


class DataSheet(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(NUM_ROWS, NUM_COLS, parent)
        self.setHorizontalHeaderLabels([col_letter(i) for i in range(NUM_COLS)])
        self.setVerticalHeaderLabels([str(i + 1) for i in range(NUM_ROWS)])

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignCenter)

        vheader = self.verticalHeader()
        vheader.setDefaultSectionSize(24)
        vheader.setMinimumSectionSize(24)

        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #c0c0c0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #e8e8e8;
                border: 1px solid #b0b0b0;
                padding: 2px;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 2px 6px;
            }
            QTableWidget::item:selected {
                background-color: #3399ff;
                color: white;
            }
        """)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self._copy()
        elif event.matches(QKeySequence.Paste):
            self._paste()
        elif event.matches(QKeySequence.Cut):
            self._cut()
        else:
            super().keyPressEvent(event)

    def _copy(self):
        selected = self.selectedIndexes()
        if not selected:
            return

        rows = sorted(set(idx.row() for idx in selected))
        cols = sorted(set(idx.column() for idx in selected))

        lines = []
        for row in rows:
            row_data = []
            for col in cols:
                idx = self.model().index(row, col)
                val = self.model().data(idx, Qt.DisplayRole)
                row_data.append(val if val else "")
            lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(lines))

    def _paste(self):
        text = QApplication.clipboard().text()
        if not text:
            return

        current = self.currentIndex()
        if not current.isValid():
            return

        start_row = current.row()
        start_col = current.column()

        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if mime.hasText():
            text = mime.text()
        else:
            return

        rows = text.split("\n")
        if rows and rows[-1] == "":
            rows = rows[:-1]

        for i, row_text in enumerate(rows):
            cells = row_text.split("\t")
            for j, cell_text in enumerate(cells):
                row = start_row + i
                col = start_col + j
                if row < NUM_ROWS and col < NUM_COLS:
                    item = QTableWidgetItem(cell_text.strip())
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.setItem(row, col, item)

    def _cut(self):
        self._copy()
        selected = self.selectedIndexes()
        for idx in selected:
            self.setItem(idx.row(), idx.column(), None)

    def get_column_data(self, col_idx):
        values = []
        for row in range(NUM_ROWS):
            item = self.item(row, col_idx)
            if item and item.text().strip():
                try:
                    values.append(float(item.text().strip()))
                except ValueError:
                    pass
        return np.array(values) if values else None

    def get_selected_column_data(self):
        selected = self.selectionModel().selectedColumns()
        if not selected:
            selected = self.selectionModel().selectedIndexes()
            if not selected:
                cols = set()
                for row in range(self.rowCount()):
                    for col in range(self.columnCount()):
                        item = self.item(row, col)
                        if item and item.text().strip():
                            cols.add(col)
                if cols:
                    col_idx = sorted(cols)[0]
                    return self.get_column_data(col_idx), col_letter(col_idx)
                return None, None

        if hasattr(selected, '__iter__'):
            col_idx = list(selected)[0].column() if hasattr(list(selected)[0], 'column') else 0
        else:
            col_idx = selected.column() if hasattr(selected, 'column') else 0

        return self.get_column_data(col_idx), col_letter(col_idx)

    def load_from_dataframe(self, df):
        max_rows = min(len(df), NUM_ROWS)
        max_cols = min(len(df.columns), NUM_COLS)

        for col_idx in range(max_cols):
            self.setHorizontalHeader(col_idx, col_letter(col_idx))
            col_name = df.columns[col_idx]
            for row_idx in range(max_rows):
                val = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row_idx, col_idx, item)

    def load_from_array(self, data, start_col=0):
        max_rows = min(len(data), NUM_ROWS)
        for row_idx in range(max_rows):
            item = QTableWidgetItem(f"{data[row_idx]:.6f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row_idx, start_col, item)


class GenerateDataDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Générer des données")
        self.resize(400, 250)

        layout = QFormLayout(self)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["Normal", "Uniforme", "Exponentielle"])
        layout.addRow("Distribution :", self.dist_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(10, NUM_ROWS)
        self.n_spin.setValue(100)
        layout.addRow("Nombre de valeurs :", self.n_spin)

        self.p1_spin = QDoubleSpinBox()
        self.p1_spin.setRange(-1e9, 1e9)
        self.p1_spin.setValue(50.0)
        layout.addRow("Moyenne (ou min) :", self.p1_spin)

        self.p2_spin = QDoubleSpinBox()
        self.p2_spin.setRange(-1e9, 1e9)
        self.p2_spin.setSingleStep(0.5)
        self.p2_spin.setValue(5.0)
        layout.addRow("Écart-type (ou max) :", self.p2_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        layout.addRow("Seed (optionnel) :", self.seed_spin)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def generate(self):
        n = self.n_spin.value()
        np.random.seed(self.seed_spin.value())
        dist = self.dist_combo.currentText()
        p1 = self.p1_spin.value()
        p2 = self.p2_spin.value()

        if dist == "Normal":
            return np.random.normal(p1, p2, n)
        elif dist == "Uniforme":
            return np.random.uniform(p1, p2, n)
        elif dist == "Exponentielle":
            return np.random.exponential(p2, n) + p1


class ColumnSelectDialog(QDialog):
    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        self.setWindowTitle("Sélectionner la colonne")
        self.resize(300, 200)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Colonne de données à analyser :"))

        self.col_combo = QComboBox()
        if columns:
            self.col_combo.addItems(columns)
        layout.addWidget(self.col_combo)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_column(self):
        return self.col_combo.currentText()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class SafeNavigationToolbar(NavigationToolbar):
    def save_figure(self, *args, **kwargs):
        try:
            file_filters = (
                "PNG Image (*.png);;"
                "PDF Document (*.pdf);;"
                "SVG Image (*.svg);;"
                "JPEG Image (*.jpg *.jpeg)"
            )
            
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Enregistrer le graphique",
                "graphique.png",
                file_filters
            )
            
            if filepath:
                self.canvas.figure.savefig(filepath)
                QMessageBox.information(self, "Succès", f"Graphique enregistré :\n{filepath}")
                
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de sauvegarder le graphique :\n{e}")


class StatisticalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StatPro - Analyse Statistique")
        self.resize(1400, 900)

        self._create_menu()
        self._create_toolbar()
        self._create_central_widget()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Prêt - Saisissez des données dans le tableur")

    def _create_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Fichier")
        file_menu.addAction("Importer CSV", self._import_csv)
        file_menu.addAction("Importer Excel", self._import_excel)
        file_menu.addSeparator()
        file_menu.addAction("Exporter résultats", self._export_results)
        file_menu.addSeparator()
        file_menu.addAction("Quitter", self.close)

        edit_menu = menubar.addMenu("Édition")
        edit_menu.addAction("Générer données exemple", self._generate_sample_data)

        help_menu = menubar.addMenu("Aide")
        help_menu.addAction("À propos", self._show_about)

    def _create_toolbar(self):
        toolbar = QToolBar("Actions rapides")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        gen_action = toolbar.addAction(" Générer données")
        gen_action.triggered.connect(self._generate_sample_data)

        import_action = toolbar.addAction(" Importer CSV")
        import_action.triggered.connect(self._import_csv)

        clear_action = toolbar.addAction(" Effacer feuille")
        clear_action.triggered.connect(self._clear_sheet)

    def _create_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_data_tab()
        self._create_capability_tab()
        self._create_normality_tab()
        self._create_outliers_tab()

    def _create_data_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Données")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(2, 2, 2, 2)

        info_label = QLabel(
            " Saisissez vos données dans le tableur ci-dessous. "
            "Sélectionnez une colonne avant d'exécuter une analyse. "
            "Les colonnes sont nommées A, B, C, ... comme dans Minitab."
        )
        info_label.setStyleSheet("background-color: #ffffcc; padding: 5px; border: 1px solid #cccc66;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.sheet = DataSheet()
        layout.addWidget(self.sheet)

        stats_group = QGroupBox("Statistiques descriptives - Colonne sélectionnée")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setFont(QFont("Courier", 10))
        stats_layout.addWidget(self.stats_text)
        layout.addWidget(stats_group)

        btn_frame = QWidget()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        refresh_btn = QPushButton(" Actualiser stats")
        refresh_btn.clicked.connect(self._update_data_display)
        clear_btn = QPushButton(" Effacer feuille")
        clear_btn.clicked.connect(self._clear_sheet)
        gen_btn = QPushButton(" Générer données")
        gen_btn.clicked.connect(self._generate_sample_data)

        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(gen_btn)
        btn_layout.addStretch()
        layout.addWidget(btn_frame)

        self.sheet.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self):
        self._update_data_display()

    def _get_active_data(self):
        data, col_name = self.sheet.get_selected_column_data()
        return data, col_name

    def _clear_sheet(self):
        self.sheet.clearContents()
        self.stats_text.clear()
        self.status_bar.showMessage("Feuille effacée")

    def _update_data_display(self):
        data, col_name = self._get_active_data()
        if data is None or len(data) == 0:
            self.stats_text.setText("Aucune donnée dans la colonne sélectionnée.")
            return

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        stats_info = (
            f"Colonne : {col_name}\n"
            f"N = {len(data)}\n"
            f"Moyenne = {mean:.6f}\n"
            f"Médiane = {np.median(data):.6f}\n"
            f"Écart-type = {std:.6f}\n"
            f"Min = {np.min(data):.6f}\n"
            f"Max = {np.max(data):.6f}\n"
            f"Asymétrie = {scipy_stats.skew(data):.4f}\n"
            f"Aplatissement = {scipy_stats.kurtosis(data):.4f}"
        )
        self.stats_text.setText(stats_info)
        self.status_bar.showMessage(f"Colonne {col_name} sélectionnée : {len(data)} valeurs")

    def _import_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Importer CSV", "", "CSV files (*.csv);;All files (*)")
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            numeric_cols = df.select_dtypes(include=[np.number])
            if len(numeric_cols.columns) == 0:
                QMessageBox.critical(self, "Erreur", "Aucune colonne numérique trouvée")
                return
            self.sheet.load_from_dataframe(numeric_cols)
            self.status_bar.showMessage(f"Données importées : {filepath}")
            self._update_data_display()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{e}")

    def _import_excel(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Importer Excel", "", "Excel files (*.xlsx *.xls);;All files (*)")
        if not filepath:
            return
        try:
            df = pd.read_excel(filepath)
            numeric_cols = df.select_dtypes(include=[np.number])
            if len(numeric_cols.columns) == 0:
                QMessageBox.critical(self, "Erreur", "Aucune colonne numérique trouvée")
                return
            self.sheet.load_from_dataframe(numeric_cols)
            self.status_bar.showMessage(f"Données importées : {filepath}")
            self._update_data_display()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{e}")

    def _generate_sample_data(self):
        dialog = GenerateDataDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            try:
                data = dialog.generate()
                col_letter_selected = "A"
                selected = self.sheet.selectionModel().selectedColumns()
                if selected:
                    col_idx = list(selected)[0].column()
                    col_letter_selected = col_letter(col_idx)

                self.sheet.load_from_array(data, start_col=ord(col_letter_selected[0]) - 65)
                dist = dialog.dist_combo.currentText()
                n = dialog.n_spin.value()
                self.status_bar.showMessage(f"Données générées en {col_letter_selected} : {dist} ({n} valeurs)")
                self._update_data_display()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

    def _get_data_from_sheet(self):
        data, col_name = self._get_active_data()
        if data is None or len(data) == 0:
            QMessageBox.warning(self, "Attention", "Aucune donnée.\nSaisissez des données dans le tableur ou sélectionnez une colonne.")
            return None, None
        return data, col_name

    def _setup_analysis_layout(self, tab):
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        left = QWidget()
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        canvas = MplCanvas(right, width=7, height=8)
        right_layout.addWidget(canvas)
        toolbar = SafeNavigationToolbar(canvas, right)
        right_layout.addWidget(toolbar)

        layout.addWidget(left)
        layout.addWidget(right, stretch=1)

        return left, left_layout, right, canvas

    def _create_capability_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Capabilité")
        left, left_layout, right, self.cap_canvas = self._setup_analysis_layout(tab)

        col_label = QLabel("Colonne à analyser : ")
        col_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(col_label)

        self.cap_col_display = QLineEdit()
        self.cap_col_display.setReadOnly(True)
        left_layout.addWidget(self.cap_col_display)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.cap_lsl = QDoubleSpinBox()
        self.cap_lsl.setRange(-1e9, 1e9)
        self.cap_lsl.setSpecialValueText("Aucune")
        params_layout.addRow("LSL (limite inférieure) :", self.cap_lsl)

        self.cap_usl = QDoubleSpinBox()
        self.cap_usl.setRange(-1e9, 1e9)
        self.cap_usl.setSpecialValueText("Aucune")
        params_layout.addRow("USL (limite supérieure) :", self.cap_usl)

        self.cap_target = QDoubleSpinBox()
        self.cap_target.setRange(-1e9, 1e9)
        self.cap_target.setSpecialValueText("Aucune")
        params_layout.addRow("Cible (Target) :", self.cap_target)

        self.cap_subgroup = QSpinBox()
        self.cap_subgroup.setRange(1, 100)
        self.cap_subgroup.setValue(1)
        params_layout.addRow("Taille sous-groupe :", self.cap_subgroup)

        self.cap_method = QComboBox()
        self.cap_method.addItems(["pooled", "rbar", "sbar"])
        self.cap_method.setCurrentText("pooled")
        params_layout.addRow("Méthode estimation σ :", self.cap_method)

        self.cap_confidence = QDoubleSpinBox()
        self.cap_confidence.setRange(80, 99.9)
        self.cap_confidence.setValue(95.0)
        self.cap_confidence.setSuffix(" %")
        params_layout.addRow("Niveau confiance :", self.cap_confidence)

        left_layout.addWidget(params_group)

        calc_btn = QPushButton(" Calculer")
        calc_btn.setMinimumHeight(40)
        calc_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        calc_btn.clicked.connect(self._run_capability)
        left_layout.addWidget(calc_btn)

        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.cap_result_text = QTextEdit()
        self.cap_result_text.setReadOnly(True)
        self.cap_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.cap_result_text)
        left_layout.addWidget(result_group)

    def _run_capability(self):
        data, col_name = self._get_data_from_sheet()
        if data is None:
            return

        self.cap_col_display.setText(f"Colonne {col_name}")

        lsl = None if self.cap_lsl.value() == self.cap_lsl.minimum() else self.cap_lsl.value()
        usl = None if self.cap_usl.value() == self.cap_usl.minimum() else self.cap_usl.value()
        target = None if self.cap_target.value() == self.cap_target.minimum() else self.cap_target.value()

        if lsl is None and usl is None:
            QMessageBox.warning(self, "Attention", "Spécifiez au moins une limite (LSL ou USL)")
            return

        try:
            subgroup = self.cap_subgroup.value()
            method = self.cap_method.currentText()
            confidence = self.cap_confidence.value() / 100.0

            analysis = CapabilityAnalysis(
                data, usl=usl, lsl=lsl, target=target,
                subgroup_size=subgroup, estimation_method=method,
                confidence_level=confidence,
            )

            results = analysis.get_results()
            summary = analysis.get_summary()

            self.cap_result_text.setText(summary)
            self._plot_capability(analysis, results)
            self.status_bar.showMessage("Analyse de capabilité terminée")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du calcul :\n{e}")

    def _plot_capability(self, analysis, results):
        self.cap_canvas.fig.clear()

        n = len(self.data) if hasattr(self, 'data') else results["n"]
        mean = results["mean"]
        sigma_within = results["std_within"]
        sigma_overall = results["std_overall"]
        lsl = results.get("lsl")
        usl = results.get("usl")

        ax1 = self.cap_canvas.fig.add_subplot(311)
        ax2 = self.cap_canvas.fig.add_subplot(312)
        ax3 = self.cap_canvas.fig.add_subplot(313)

        raw_data = self._get_raw_data_for_plot()
        if raw_data is None:
            return

        x = np.linspace(np.min(raw_data) - 3 * sigma_overall, np.max(raw_data) + 3 * sigma_overall, 200)
        y_within = scipy_stats.norm.pdf(x, mean, sigma_within)
        y_overall = scipy_stats.norm.pdf(x, mean, sigma_overall)

        bins = max(10, int(np.sqrt(len(raw_data))))
        ax1.hist(raw_data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black", label="Histogramme")
        ax1.plot(x, y_within, "r-", linewidth=2, label=f"Courbe intra (σ={sigma_within:.4f})")
        ax1.plot(x, y_overall, "g--", linewidth=2, label=f"Courbe globale (σ={sigma_overall:.4f})")

        if lsl is not None:
            ax1.axvline(lsl, color="red", linestyle="-", linewidth=2, label=f"LSL={lsl}")
        if usl is not None:
            ax1.axvline(usl, color="red", linestyle="-", linewidth=2, label=f"USL={usl}")
        if results.get("target") is not None:
            ax1.axvline(results["target"], color="green", linestyle=":", linewidth=2, label=f"Cible={results['target']}")

        ax1.axvline(mean, color="blue", linestyle="--", linewidth=1.5, label=f"Moy={mean:.4f}")
        ax1.legend(fontsize=7)
        ax1.set_title("Histogramme et courbes normales")
        ax1.set_ylabel("Densité")

        if lsl is not None and usl is not None:
            ax2.hist(raw_data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")
            ax2.fill_between(x, y_within, where=(x < lsl) | (x > usl), alpha=0.3, color="red", label="Non-conforme")
            ax2.fill_between(x, y_within, where=(x >= lsl) & (x <= usl), alpha=0.3, color="green", label="Conforme")
            ax2.plot(x, y_within, "r-", linewidth=2)
            if lsl is not None:
                ax2.axvline(lsl, color="red", linestyle="-", linewidth=2)
            if usl is not None:
                ax2.axvline(usl, color="red", linestyle="-", linewidth=2)
            ax2.legend(fontsize=7)
            ax2.set_title("Zone conforme / non-conforme")
        else:
            ax2.hist(raw_data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")
            ax2.plot(x, y_within, "r-", linewidth=2)
            if lsl is not None:
                ax2.axvline(lsl, color="red", linestyle="-", linewidth=2)
                ax2.fill_between(x, y_within, where=(x < lsl), alpha=0.3, color="red")
            if usl is not None:
                ax2.axvline(usl, color="red", linestyle="-", linewidth=2)
                ax2.fill_between(x, y_within, where=(x > usl), alpha=0.3, color="red")
            ax2.set_title("Zone non-conforme")

        scipy_stats.probplot(raw_data, dist="norm", plot=ax3)
        ax3.set_title("QQ Plot")
        ax3.grid(True, alpha=0.3)

        try:
            self.cap_canvas.fig.tight_layout()
        except Exception:
            pass
        self.cap_canvas.draw()

    def _get_raw_data_for_plot(self):
        data, _ = self._get_data_from_sheet()
        return data

    def _create_normality_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Normalité")
        left, left_layout, right, self.norm_canvas = self._setup_analysis_layout(tab)

        col_label = QLabel("Colonne à analyser : ")
        col_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(col_label)

        self.norm_col_display = QLineEdit()
        self.norm_col_display.setReadOnly(True)
        left_layout.addWidget(self.norm_col_display)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.norm_alpha = QDoubleSpinBox()
        self.norm_alpha.setRange(0.001, 0.5)
        self.norm_alpha.setSingleStep(0.005)
        self.norm_alpha.setValue(0.05)
        params_layout.addRow("Seuil α :", self.norm_alpha)

        tests_group = QGroupBox("Tests à effectuer")
        tests_layout = QVBoxLayout(tests_group)

        self.norm_tests = {}
        test_names = ["Shapiro-Wilk", "Anderson-Darling", "Khi-deux (χ²)",
                       "D'Agostino-Pearson", "Jarque-Bera", "Ryan-Joiner"]
        for name in test_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.norm_tests[name] = cb
            tests_layout.addWidget(cb)

        params_layout.addRow(tests_group)
        left_layout.addWidget(params_group)

        run_btn = QPushButton(" Exécuter tests")
        run_btn.setMinimumHeight(40)
        run_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        run_btn.clicked.connect(self._run_normality)
        left_layout.addWidget(run_btn)

        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.norm_result_text = QTextEdit()
        self.norm_result_text.setReadOnly(True)
        self.norm_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.norm_result_text)
        left_layout.addWidget(result_group)

    def _run_normality(self):
        data, col_name = self._get_data_from_sheet()
        if data is None:
            return

        self.norm_col_display.setText(f"Colonne {col_name}")

        try:
            alpha = self.norm_alpha.value()
            tests = NormalityTests(data, alpha=alpha)
            full_results = tests.get_results()

            summary_lines = ["=" * 60, "TESTS DE NORMALITÉ", "=" * 60]
            summary_lines.append(f"\nColonne : {col_name}")
            summary_lines.append(f"N = {full_results['n']}")
            summary_lines.append(f"Moyenne = {full_results['mean']:.6f}")
            summary_lines.append(f"Écart-type = {full_results['std']:.6f}")
            summary_lines.append(f"Asymétrie = {full_results['skewness']:.4f}")
            summary_lines.append(f"Aplatissement = {full_results['kurtosis']:.4f}")
            summary_lines.append(f"\nSeuil α = {full_results['alpha']}")
            summary_lines.append("-" * 60)

            for name, cb in self.norm_tests.items():
                if cb.isChecked() and name in full_results["tests"]:
                    result = full_results["tests"][name]
                    summary_lines.append(f"\n{name} :")
                    summary_lines.append(f"  Statistique = {result['statistic']:.6f}")
                    if result.get("p_value") is not None:
                        summary_lines.append(f"  p-value = {result['p_value']:.6f}")
                        normal_str = "OUI" if result["normal"] else "NON"
                        summary_lines.append(f"  Normal = {normal_str}")
                    if result.get("df") is not None:
                        summary_lines.append(f"  Degrés de liberté = {result['df']}")
                    if "critical_value" in result and result["critical_value"] is not None:
                        summary_lines.append(f"  Valeur critique (α={alpha}) = {result['critical_value']:.6f}")
                    if "critical_95" in result:
                        summary_lines.append(f"  Critique (95%) = {result['critical_95']:.6f}")
                        normal_str = "OUI" if result["normal"] else "NON"
                        summary_lines.append(f"  Normal = {normal_str}")

            summary_lines.append("=" * 60)
            self.norm_result_text.setText("\n".join(summary_lines))

            self._plot_normality(data)
            self.status_bar.showMessage("Tests de normalité terminés")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors des tests :\n{e}")

    def _plot_normality(self, data):
        self.norm_canvas.fig.clear()

        ax1 = self.norm_canvas.fig.add_subplot(221)
        ax2 = self.norm_canvas.fig.add_subplot(222)
        ax3 = self.norm_canvas.fig.add_subplot(223)
        ax4 = self.norm_canvas.fig.add_subplot(224)

        bins = max(10, int(np.sqrt(len(data))))
        ax1.hist(data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")
        x = np.linspace(np.min(data), np.max(data), 100)
        ax1.plot(x, scipy_stats.norm.pdf(x, np.mean(data), np.std(data, ddof=1)), "r-", linewidth=2)
        ax1.set_title("Histogramme avec courbe normale")

        scipy_stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title("QQ Plot")
        ax2.grid(True, alpha=0.3)

        ax3.boxplot(data, vert=True)
        ax3.set_title("Boîte à moustaches")

        ax4.hist(data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_kde = np.linspace(np.min(data), np.max(data), 200)
        ax4.plot(x_kde, kde(x_kde), "g-", linewidth=2, label="KDE")
        ax4.plot(x_kde, scipy_stats.norm.pdf(x_kde, np.mean(data), np.std(data, ddof=1)), "r--", linewidth=2, label="Normal")
        ax4.legend(fontsize=8)
        ax4.set_title("Densité KDE vs Normale")

        try:
            self.norm_canvas.fig.tight_layout()
        except Exception:
            pass
        self.norm_canvas.draw()

    def _create_outliers_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Valeurs aberrantes")
        left, left_layout, right, self.out_canvas = self._setup_analysis_layout(tab)

        col_label = QLabel("Colonne à analyser : ")
        col_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(col_label)

        self.out_col_display = QLineEdit()
        self.out_col_display.setReadOnly(True)
        left_layout.addWidget(self.out_col_display)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.out_alpha = QDoubleSpinBox()
        self.out_alpha.setRange(0.001, 0.5)
        self.out_alpha.setSingleStep(0.005)
        self.out_alpha.setValue(0.05)
        params_layout.addRow("Seuil α :", self.out_alpha)

        tests_group = QGroupBox("Méthodes")
        tests_layout = QVBoxLayout(tests_group)

        self.out_tests = {}
        test_names = ["IQR", "Z-score", "Z-score modifié", "Grubbs", "Dixon", "Chauvenet", "Tukey (extrêmes)"]
        for name in test_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.out_tests[name] = cb
            tests_layout.addWidget(cb)

        params_layout.addRow(tests_group)
        left_layout.addWidget(params_group)

        detect_btn = QPushButton(" Détecter")
        detect_btn.setMinimumHeight(40)
        detect_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        detect_btn.clicked.connect(self._run_outliers)
        left_layout.addWidget(detect_btn)

        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.out_result_text = QTextEdit()
        self.out_result_text.setReadOnly(True)
        self.out_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.out_result_text)
        left_layout.addWidget(result_group)

    def _run_outliers(self):
        data, col_name = self._get_data_from_sheet()
        if data is None:
            return

        self.out_col_display.setText(f"Colonne {col_name}")

        try:
            alpha = self.out_alpha.value()
            detector = OutlierDetection(data, alpha=alpha)
            full_results = detector.get_results()

            summary_lines = ["=" * 60, "DÉTECTION DE VALEURS ABERRANTES", "=" * 60]
            summary_lines.append(f"\nColonne : {col_name}")
            summary_lines.append(f"N = {full_results['n']}")
            summary_lines.append(f"Moyenne = {full_results['mean']:.6f}")
            summary_lines.append(f"Médiane = {full_results['median']:.6f}")
            summary_lines.append(f"Écart-type = {full_results['std']:.6f}")
            summary_lines.append(f"Q1 = {full_results['q1']:.6f}")
            summary_lines.append(f"Q3 = {full_results['q3']:.6f}")
            summary_lines.append(f"IQR = {full_results['iqr']:.6f}")
            summary_lines.append(f"\nSeuil α = {full_results['alpha']}")
            summary_lines.append("-" * 60)

            method_map = {
                "IQR": "IQR (1.5×IQR)",
                "Z-score": "Z-score",
                "Z-score modifié": "Z-score modifié (MAD)",
                "Grubbs": "Grubbs",
                "Dixon": "Dixon",
                "Chauvenet": "Chauvenet",
                "Tukey (extrêmes)": "Tukey (3×IQR, extrêmes)",
            }

            for display_name, cb in self.out_tests.items():
                if cb.isChecked():
                    test_name = method_map.get(display_name, display_name)
                    if test_name in full_results["tests"]:
                        result = full_results["tests"][test_name]
                        summary_lines.append(f"\n{test_name} :")
                        summary_lines.append(f"  Valeurs détectées : {result['outlier_count']}")
                        if result["outlier_count"] > 0:
                            for val, idx in zip(result["outlier_values"], result["outlier_indices"]):
                                summary_lines.append(f"    [{idx}] Valeur = {val:.6f}")
                        else:
                            summary_lines.append(f"    Aucune valeur aberrante détectée")

            summary_lines.append("=" * 60)
            self.out_result_text.setText("\n".join(summary_lines))

            self._plot_outliers(full_results, detector)
            self.status_bar.showMessage("Détection de valeurs aberrantes terminée")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la détection :\n{e}")

    def _plot_outliers(self, full_results, detector):
        self.out_canvas.fig.clear()

        data, _ = self._get_data_from_sheet()

        ax1 = self.out_canvas.fig.add_subplot(311)
        ax2 = self.out_canvas.fig.add_subplot(312)
        ax3 = self.out_canvas.fig.add_subplot(313)

        all_outlier_indices = detector.get_outlier_indices()

        colors = ["red" if i in all_outlier_indices else "blue" for i in range(len(data))]

        ax1.scatter(range(len(data)), data, c=colors, alpha=0.6, s=10)
        ax1.axhline(np.mean(data), color="green", linestyle="--", label=f"Moy={np.mean(data):.2f}")
        ax1.axhline(np.mean(data) + 3 * np.std(data, ddof=1), color="red", linestyle=":", alpha=0.5)
        ax1.axhline(np.mean(data) - 3 * np.std(data, ddof=1), color="red", linestyle=":", alpha=0.5)
        ax1.legend(fontsize=7)
        ax1.set_title("Graphique des données (valeurs aberrantes en rouge)")
        ax1.set_ylabel("Valeur")

        bp = ax2.boxplot(data, vert=True, patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor("lightblue")
        ax2.set_title("Boîte à moustaches avec valeurs aberrantes")

        deviations = np.abs(data - np.median(data))
        ax3.scatter(range(len(data)), deviations, alpha=0.6, color="blue", s=10)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad > 0:
            threshold = 3.5 / 0.6745 * mad
            ax3.axhline(threshold, color="red", linestyle="--", label="Seuil (3.5 MAD)")
            ax3.legend(fontsize=7)
        ax3.set_title("Écarts à la médiane")
        ax3.set_xlabel("Index")
        ax3.set_ylabel("Écart absolu")

        try:
            self.out_canvas.fig.tight_layout()
        except Exception:
            pass
        self.out_canvas.draw()

    def _export_results(self):
        texts = []
        if self.cap_result_text.toPlainText().strip():
            texts.append(("CAPABILITÉ", self.cap_result_text.toPlainText()))
        if self.norm_result_text.toPlainText().strip():
            texts.append(("NORMALITÉ", self.norm_result_text.toPlainText()))
        if self.out_result_text.toPlainText().strip():
            texts.append(("VALEURS ABERRANTES", self.out_result_text.toPlainText()))

        if not texts:
            QMessageBox.warning(self, "Attention", "Aucun résultat à exporter")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Exporter résultats", "", "Text files (*.txt);;All files (*)")
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("RÉSULTATS D'ANALYSE STATISTIQUE\n")
                f.write("=" * 60 + "\n\n")
                for title, content in texts:
                    f.write(f"{title}\n")
                    f.write(content)
                    f.write("\n\n")
            self.status_bar.showMessage(f"Résultats exportés : {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'exporter :\n{e}")

    def _show_about(self):
        QMessageBox.information(
            self, "À propos",
            "StatPro - Analyse Statistique\n\n"
            "Application d'analyse statistique\n"
            "Alternative à Minitab\n\n"
            "Fonctionnalités :\n"
            "- Tableur avec colonnes A, B, C, ...\n"
            "- Analyse de capabilité (Cp, Cpk, Pp, Ppk, Cpm)\n"
            "- Tests de normalité (Shapiro-Wilk, Anderson-Darling, Khi-deux, etc.)\n"
            "- Détection de valeurs aberrantes (Grubbs, Dixon, IQR, etc.)\n\n"
            "© 2026",
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StatisticalApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
