import sys
import os
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
from stats_engine.correlation import CorrelationMatrix
from stats_engine.probability_plots import ProbabilityPlot
from stats_engine.msa import GageRRStudy
from stats_engine.export import ReportExporter


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

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self.setAlternatingRowColors(False)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #c0c0c0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #e8e8e8;
                border: 1px solid #b0b0b0;
                padding: 2px;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 2px 6px;
                color: #1a1a1a;
                background-color: white;
            }
            QTableWidget::item:selected {
                background-color: #3399ff;
                color: white;
            }
        """)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        
        cut_action = menu.addAction(" Couper")
        cut_action.triggered.connect(self._cut)
        
        copy_action = menu.addAction(" Copier")
        copy_action.triggered.connect(self._copy)
        
        paste_action = menu.addAction(" Coller")
        paste_action.triggered.connect(self._paste)
        
        menu.addSeparator()
        
        clear_action = menu.addAction(" Effacer cellules")
        clear_action.triggered.connect(self._clear_cells)
        
        fill_action = menu.addAction(" Remplir avec séquence...")
        fill_action.triggered.connect(self._fill_sequence)
        
        menu.addSeparator()
        
        insert_row_action = menu.addAction(" Insérer ligne")
        insert_row_action.triggered.connect(self._insert_row)
        
        delete_row_action = menu.addAction(" Supprimer ligne")
        delete_row_action.triggered.connect(self._delete_row)
        
        menu.addSeparator()
        
        stats_action = menu.addAction(" Statistiques rapides...")
        stats_action.triggered.connect(self._quick_stats)
        
        menu.exec_(self.mapToGlobal(pos))

    def _clear_cells(self):
        selected = self.selectedIndexes()
        for idx in selected:
            self.setItem(idx.row(), idx.column(), None)

    def _fill_sequence(self):
        selected = self.selectedIndexes()
        if not selected:
            return
        rows = sorted(set(idx.row() for idx in selected))
        cols = sorted(set(idx.column() for idx in selected))
        if not rows or not cols:
            return
        
        from PyQt5.QtWidgets import QInputDialog
        start, ok = QInputDialog.getDouble(self, "Remplir séquence", "Valeur de départ :", 1.0, -1e9, 1e9, 6)
        if not ok:
            return
        step, ok2 = QInputDialog.getDouble(self, "Remplir séquence", "Incrément :", 1.0, -1e9, 1e9, 6)
        if not ok2:
            return
        
        val = start
        for row in rows:
            for col in cols:
                item = QTableWidgetItem(f"{val:.6f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row, col, item)
                val += step

    def _insert_row(self):
        current = self.currentIndex()
        if not current.isValid():
            return
        row = current.row()
        self.insertRow(row)
        new_label = [str(i + 1) for i in range(self.rowCount())]
        self.setVerticalHeaderLabels(new_label)

    def _delete_row(self):
        current = self.currentIndex()
        if not current.isValid():
            return
        row = current.row()
        self.removeRow(row)
        new_label = [str(i + 1) for i in range(self.rowCount())]
        self.setVerticalHeaderLabels(new_label)

    def _quick_stats(self):
        selected = self.selectedIndexes()
        if not selected:
            return
        rows = sorted(set(idx.row() for idx in selected))
        cols = sorted(set(idx.column() for idx in selected))
        values = []
        for row in rows:
            for col in cols:
                item = self.item(row, col)
                if item and item.text().strip():
                    try:
                        values.append(float(item.text().strip()))
                    except ValueError:
                        pass
        if not values:
            QMessageBox.information(self, "Statistiques rapides", "Aucune valeur numérique dans la sélection.")
            return
        arr = np.array(values)
        msg = (f"Sélection : {len(arr)} valeurs\n\n"
               f"Moyenne = {np.mean(arr):.6f}\n"
               f"Médiane = {np.median(arr):.6f}\n"
               f"Écart-type = {np.std(arr, ddof=1):.6f}\n"
               f"Min = {np.min(arr):.6f}\n"
               f"Max = {np.max(arr):.6f}\n"
               f"Somme = {np.sum(arr):.6f}")
        QMessageBox.information(self, "Statistiques rapides", msg)

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
        
        self.cap_canvas = None
        self.norm_canvas = None
        self.out_canvas = None
        self.dist_canvas = None
        self.corr_canvas = None
        self.reg_canvas = None
        self.tt_canvas = None
        self.anova_canvas = None
        self.boxplot_canvas = None
        self.cc_canvas = None
        self.prob_canvas = None
        self.msa_canvas = None
        
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
        self._create_distribution_tab()
        self._create_outliers_tab()
        self._create_normality_tab()
        self._create_capability_tab()
        self._create_correlation_tab()
        self._create_regression_tab()
        self._create_ttest_tab()
        self._create_anova_tab()
        self._create_boxplot_tab()
        self._create_control_charts_tab()
        self._create_probplot_tab()
        self._create_msa_tab()

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._refresh_all_combos()

    def _on_tab_changed(self, index):
        self._refresh_all_combos()
        if self.tabs.tabText(index) == "ANOVA":
            self._update_anova_combos()
        elif self.tabs.tabText(index) == "Boxplots":
            self._update_boxplot_combos()

    def _refresh_all_combos(self):
        combos = []
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, QComboBox) and attr.endswith('_col_combo'):
                combos.append(obj)
            if isinstance(obj, QComboBox) and attr in ('reg_y_combo', 'reg_x_combo', 'tt_col1_combo', 'tt_col2_combo',
                                                       'msa_part_combo', 'msa_op_combo', 'msa_meas_combo'):
                combos.append(obj)
        seen = set()
        unique = []
        for c in combos:
            if id(c) not in seen:
                seen.add(id(c))
                unique.append(c)
        cols = []
        for i in range(self.sheet.columnCount()):
            data = self.sheet.get_column_data(i)
            if data is not None and len(data) > 0:
                cols.append(col_letter(i))
        for c in unique:
            c.clear()
            c.addItems(cols if cols else ["(aucune donnée)"])

        if hasattr(self, 'anova_col_combos') and self.anova_col_combos:
            self._update_anova_combos()
        if hasattr(self, 'boxplot_col_combos') and self.boxplot_col_combos:
            self._update_boxplot_combos()
        if not cols:
            cols = ['(aucune donnée)']
        for combo in unique:
            current = combo.currentText()
            combo.clear()
            combo.addItems(cols)
            if current in cols:
                combo.setCurrentText(current)

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

    def _create_distribution_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Identification distribution")
        left, left_layout, right, self.dist_canvas = self._setup_analysis_layout(tab)

        col_row = QHBoxLayout()
        self.dist_col_combo = QComboBox()
        self.dist_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.dist_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.dist_alpha = QDoubleSpinBox()
        self.dist_alpha.setRange(0.001, 0.5)
        self.dist_alpha.setSingleStep(0.005)
        self.dist_alpha.setValue(0.05)
        self._tip(self.dist_alpha, "Seuil de significativité α. Si p-value > α, la distribution est considérée comme un bon ajustement aux données.")
        params_layout.addRow("Seuil α :", self.dist_alpha)

        dists_group = QGroupBox("Distributions à tester")
        dists_layout = QVBoxLayout(dists_group)

        self.dist_checks = {}
        dist_desc = [
            ("Normal", "Loi gaussienne. Symétrique, définie par moyenne et écart-type. La plus courante."),
            ("Log-Normale", "Le logarithme des données suit une loi normale. Pour données strictement positives et asymétriques."),
            ("Weibull (2P)", "Flexible pour fiabilité/durée de vie. Shape + scale. Inclut exponentielle comme cas particulier."),
            ("Exponentielle", "Modélise les temps entre événements. Cas particulier de Weibull avec shape=1."),
            ("Gamma", "Généralisation de l'exponentielle. Pour temps d'attente de k événements."),
            ("Logistique", "Similaire à normale mais avec queues plus lourdes."),
            ("Gumbel (max)", "Pour valeurs extrêmes (maxima). Utilisée en hydrologie et fiabilité."),
            ("Cauchy", "Queue très lourde, pas de moyenne ni variance définies. Pour données avec outliers fréquents."),
            ("Rayleigh", "Pour magnitude de vecteurs 2D. Cas particulier de Weibull."),
            ("Uniforme", "Toutes les valeurs equally probables entre min et max."),
            ("Student-t", "Similaire à normale mais queue plus lourde. Pour petits échantillons."),
            ("Laplace", "Double exponentielle. Plus de kurtosis que la normale. Pour données avec pics centraux."),
        ]
        for name, desc in dist_desc:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setToolTip(desc)
            self.dist_checks[name] = cb
            dists_layout.addWidget(cb)

        params_layout.addRow(dists_group)
        left_layout.addWidget(params_group)

        calc_btn = QPushButton(" Identifier")
        calc_btn.setMinimumHeight(40)
        calc_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        calc_btn.clicked.connect(self._run_distribution)
        left_layout.addWidget(calc_btn)

        result_group = QGroupBox("Résultats (classés par p-value)")
        result_layout = QVBoxLayout(result_group)
        self.dist_result_text = QTextEdit()
        self.dist_result_text.setReadOnly(True)
        self.dist_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.dist_result_text)
        left_layout.addWidget(result_group)

    def _run_distribution(self):
        data, col_name = self._get_combo_data(self.dist_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return

        try:
            alpha = self.dist_alpha.value()

            dist_map = {
                "Normal": ("norm", lambda d: scipy_stats.norm.fit(d)),
                "Log-Normale": ("lognorm", lambda d: scipy_stats.lognorm.fit(d, floc=0)),
                "Weibull (2P)": ("weibull_min", lambda d: scipy_stats.weibull_min.fit(d, floc=0)),
                "Exponentielle": ("expon", lambda d: scipy_stats.expon.fit(d)),
                "Gamma": ("gamma", lambda d: scipy_stats.gamma.fit(d, floc=0)),
                "Logistique": ("logistic", lambda d: scipy_stats.logistic.fit(d)),
                "Gumbel (max)": ("gumbel_r", lambda d: scipy_stats.gumbel_r.fit(d)),
                "Cauchy": ("cauchy", lambda d: scipy_stats.cauchy.fit(d)),
                "Rayleigh": ("rayleigh", lambda d: scipy_stats.rayleigh.fit(d, floc=0)),
                "Uniforme": ("uniform", lambda d: scipy_stats.uniform.fit(d)),
                "Student-t": ("t", lambda d: scipy_stats.t.fit(d)),
                "Laplace": ("laplace", lambda d: scipy_stats.laplace.fit(d)),
            }

            results = []
            for display_name, cb in self.dist_checks.items():
                if cb.isChecked() and display_name in dist_map:
                    dist_name, fit_func = dist_map[display_name]
                    try:
                        params = fit_func(data)
                        loc = params[-2] if len(params) >= 2 else 0
                        scale = params[-1] if len(params) >= 1 else 1
                        shape_params = params[:-2] if len(params) > 2 else ()

                        ks_stat, ks_pvalue = scipy_stats.kstest(
                            data, dist_name, args=params
                        )
                        results.append({
                            "name": display_name,
                            "dist_name": dist_name,
                            "params": params,
                            "ks_stat": ks_stat,
                            "p_value": ks_pvalue,
                            "passes": ks_pvalue > alpha,
                        })
                    except Exception:
                        results.append({
                            "name": display_name,
                            "dist_name": dist_name,
                            "params": None,
                            "ks_stat": None,
                            "p_value": None,
                            "passes": False,
                            "error": True,
                        })

            results.sort(key=lambda r: r["p_value"] if r["p_value"] is not None else -1, reverse=True)

            lines = ["=" * 60, "IDENTIFICATION DE DISTRIBUTION", "=" * 60]
            lines.append(f"\nColonne : {col_name}")
            lines.append(f"N = {len(data)}")
            lines.append(f"Moyenne = {np.mean(data):.6f}")
            lines.append(f"Écart-type = {np.std(data, ddof=1):.6f}")
            lines.append(f"Seuil α = {alpha}")
            lines.append("\n" + "-" * 60)
            lines.append(f"{'Rang':<6} {'Distribution':<18} {'KS Stat':<12} {'p-value':<12} {'Ajusté?':<10}")
            lines.append("-" * 60)

            best_dist = results[0] if results else None
            for i, r in enumerate(results):
                if r.get("error"):
                    lines.append(f"{i+1:<6} {r['name']:<18} {'ERREUR':<12} {'-':<12} {'NON':<10}")
                else:
                    normal_str = "OUI" if r["passes"] else "NON"
                    lines.append(f"{i+1:<6} {r['name']:<18} {r['ks_stat']:<12.6f} {r['p_value']:<12.6f} {normal_str:<10}")

            lines.append("-" * 60)
            if best_dist and not best_dist.get("error"):
                lines.append(f"\nMeilleure distribution : {best_dist['name']}")
                lines.append(f"  p-value = {best_dist['p_value']:.6f}")
                lines.append(f"  Statistique KS = {best_dist['ks_stat']:.6f}")
                if best_dist["params"] is not None:
                    lines.append(f"  Paramètres : {best_dist['params']}")
            lines.append("=" * 60)

            self.dist_result_text.setText("\n".join(lines))
            self._plot_distribution(data, results)
            self.status_bar.showMessage("Identification de distribution terminée")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'identification :\n{e}")

    def _plot_distribution(self, data, results):
        self.dist_canvas.fig.clear()

        valid_results = [r for r in results if r["p_value"] is not None and not r.get("error")]
        n = len(valid_results)
        if n == 0:
            ax = self.dist_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Aucune distribution valide", ha="center", va="center", fontsize=14)
            self.dist_canvas.draw()
            return

        cols = 3
        rows = (n + cols - 1) // cols

        self.dist_canvas.fig.set_size_inches(14, rows * 5)

        gs = self.dist_canvas.fig.add_gridspec(
            rows * 2, cols,
            left=0.06, right=0.96, top=0.96, bottom=0.04,
            wspace=0.30, hspace=0.50,
        )

        data_min = np.min(data)
        data_max = np.max(data)

        for i, r in enumerate(valid_results):
            row = i // cols
            col = i % cols

            ax_hist = self.dist_canvas.fig.add_subplot(gs[row * 2, col])
            ax_prob = self.dist_canvas.fig.add_subplot(gs[row * 2 + 1, col])

            bins = max(10, int(np.sqrt(len(data))))
            ax_hist.hist(data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")

            if r["params"] is not None:
                x = np.linspace(data_min, data_max, 300)
                dist_obj = getattr(scipy_stats, r["dist_name"])
                pdf_vals = dist_obj.pdf(x, *r["params"])
                ax_hist.plot(x, pdf_vals, "r-", linewidth=2)

            ax_hist.set_title(f"{r['name']}  (KS={r['ks_stat']:.4f}, p={r['p_value']:.4f})")
            ax_hist.grid(True, alpha=0.3)

            try:
                if r["dist_name"] == "lognormal":
                    valid = data[data > 0]
                    scipy_stats.probplot(np.log(valid), dist="norm", plot=ax_prob)
                else:
                    scipy_stats.probplot(data, dist=r["dist_name"], plot=ax_prob)
            except Exception:
                ax_prob.text(0.5, 0.5, "ProbPlot indisponible", ha="center", va="center", fontsize=8)
            ax_prob.grid(True, alpha=0.3)

        try:
            self.dist_canvas.fig.tight_layout()
        except Exception:
            pass
        self.dist_canvas.draw()

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

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(500)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        canvas = MplCanvas(right, width=7, height=8)
        right_layout.addWidget(canvas)
        toolbar = SafeNavigationToolbar(canvas, right)
        right_layout.addWidget(toolbar)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 700])
        layout.addWidget(splitter)

        return left, left_layout, right, canvas

    def _create_capability_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Capabilité")
        left, left_layout, right, self.cap_canvas = self._setup_analysis_layout(tab)

        col_row = QHBoxLayout()
        self.cap_col_combo = QComboBox()
        self.cap_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.cap_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.cap_lsl = QDoubleSpinBox()
        self.cap_lsl.setRange(-1e9, 1e9)
        self.cap_lsl.setSpecialValueText("Aucune")
        self._tip(self.cap_lsl, "Limite inférieure de spécification (LSL). Valeur minimale acceptable du processus. Si aucune, l'analyse est unilatérale supérieure.")
        params_layout.addRow("LSL (limite inférieure) :", self.cap_lsl)

        self.cap_usl = QDoubleSpinBox()
        self.cap_usl.setRange(-1e9, 1e9)
        self.cap_usl.setSpecialValueText("Aucune")
        self._tip(self.cap_usl, "Limite supérieure de spécification (USL). Valeur maximale acceptable du processus. Si aucune, l'analyse est unilatérale inférieure.")
        params_layout.addRow("USL (limite supérieure) :", self.cap_usl)

        self.cap_target = QDoubleSpinBox()
        self.cap_target.setRange(-1e9, 1e9)
        self.cap_target.setSpecialValueText("Aucune")
        self._tip(self.cap_target, "Valeur cible idéale du processus. Utilisée pour calculer Cpm, qui pénalise les écarts à la cible même dans les limites.")
        params_layout.addRow("Cible (Target) :", self.cap_target)

        self.cap_subgroup = QSpinBox()
        self.cap_subgroup.setRange(1, 100)
        self.cap_subgroup.setValue(1)
        self._tip(self.cap_subgroup, "Taille du sous-groupe pour l'estimation σ intra. 1 = individus (MR). >1 = sous-groupes rationnels.")
        params_layout.addRow("Taille sous-groupe :", self.cap_subgroup)

        self.cap_method = QComboBox()
        self.cap_method.addItems(["pooled", "rbar", "sbar"])
        self.cap_method.setCurrentText("pooled")
        self._tip(self.cap_method, "Méthode d'estimation de σ intra : pooled (écart-type groupé), rbar (R̄/d2), sbar (S̄/c4). Pooled est recommandé pour sous-groupes inégaux.")
        params_layout.addRow("Méthode estimation σ :", self.cap_method)

        self.cap_dist = QComboBox()
        self.cap_dist.addItems([
            "Normal", "Log-Normale", "Weibull (2P)", "Exponentielle",
            "Gamma", "Logistique", "Gumbel (max)", "Cauchy",
            "Rayleigh", "Uniforme", "Student-t", "Laplace",
        ])
        self._tip(self.cap_dist, "Distribution supposée des données. Normal = indices classiques. Autres = indices basés sur les percentiles (Pp/Ppk).")
        params_layout.addRow("Distribution :", self.cap_dist)

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
        data, col_name = self._get_combo_data(self.cap_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return

        lsl = None if self.cap_lsl.value() == self.cap_lsl.minimum() else self.cap_lsl.value()
        usl = None if self.cap_usl.value() == self.cap_usl.minimum() else self.cap_usl.value()
        target = None if self.cap_target.value() == self.cap_target.minimum() else self.cap_target.value()

        if lsl is None and usl is None:
            QMessageBox.warning(self, "Attention", "Spécifiez au moins une limite (LSL ou USL)")
            return

        try:
            subgroup = self.cap_subgroup.value()
            method = self.cap_method.currentText()
            distribution = self.cap_dist.currentText()
            is_normal = (distribution == "Normal")

            analysis = CapabilityAnalysis(data, usl=usl, lsl=lsl, target=target,
                                         subgroup_size=subgroup, estimation_method=method)
            results = analysis.get_results()

            non_normal_info = None
            if not is_normal:
                dist_map = {
                    "Log-Normale": ("lognorm", lambda d: scipy_stats.lognorm.fit(d, floc=0)),
                    "Weibull (2P)": ("weibull_min", lambda d: scipy_stats.weibull_min.fit(d, floc=0)),
                    "Exponentielle": ("expon", lambda d: scipy_stats.expon.fit(d)),
                    "Gamma": ("gamma", lambda d: scipy_stats.gamma.fit(d, floc=0)),
                    "Logistique": ("logistic", lambda d: scipy_stats.logistic.fit(d)),
                    "Gumbel (max)": ("gumbel_r", lambda d: scipy_stats.gumbel_r.fit(d)),
                    "Cauchy": ("cauchy", lambda d: scipy_stats.cauchy.fit(d)),
                    "Rayleigh": ("rayleigh", lambda d: scipy_stats.rayleigh.fit(d, floc=0)),
                    "Uniforme": ("uniform", lambda d: scipy_stats.uniform.fit(d)),
                    "Student-t": ("t", lambda d: scipy_stats.t.fit(d)),
                    "Laplace": ("laplace", lambda d: scipy_stats.laplace.fit(d)),
                }
                dist_info = dist_map.get(distribution)
                if dist_info:
                    dist_name, fit_func = dist_info
                    params = fit_func(data)
                    dist_obj = getattr(scipy_stats, dist_name)
                    p00135 = dist_obj.ppf(0.00135, *params)
                    p50 = dist_obj.ppf(0.5, *params)
                    p99865 = dist_obj.ppf(0.99865, *params)

                if usl is not None and lsl is not None:
                    pp_nn = (usl - lsl) / (p99865 - p00135)
                    ppu_nn = (usl - p50) / (p99865 - p50)
                    ppl_nn = (p50 - lsl) / (p50 - p00135)
                    ppk_nn = min(ppu_nn, ppl_nn)
                elif usl is not None:
                    pp_nn = None
                    ppl_nn = None
                    ppu_nn = (usl - p50) / (p99865 - p50)
                    ppk_nn = ppu_nn
                else:
                    pp_nn = None
                    ppu_nn = None
                    ppl_nn = (p50 - lsl) / (p50 - p00135)
                    ppk_nn = ppl_nn

                non_normal_info = {
                    "distribution": distribution,
                    "dist_name": dist_name,
                    "params": params,
                    "p00135": p00135,
                    "p50": p50,
                    "p99865": p99865,
                    "pp": pp_nn,
                    "ppu": ppu_nn,
                    "ppl": ppl_nn,
                    "ppk": ppk_nn,
                }
                results["pp_non_normal"] = pp_nn
                results["ppu_non_normal"] = ppu_nn
                results["ppl_non_normal"] = ppl_nn
                results["ppk_non_normal"] = ppk_nn

            lines = ["=" * 60, "ANALYSE DE CAPABILITÉ", "=" * 60]
            lines.append(f"\nColonne : {col_name}")
            lines.append(f"Distribution : {distribution}")
            lines.append(f"N = {results['n']}")
            lines.append(f"Moyenne = {results['mean']:.6f}")
            lines.append(f"Écart-type (intra) = {results['std_within']:.6f}")
            lines.append(f"Écart-type (global) = {results['std_overall']:.6f}")
            lines.append(f"\nLimites :")
            if lsl is not None:
                lines.append(f"  LSL = {lsl:.6f}")
            if usl is not None:
                lines.append(f"  USL = {usl:.6f}")
            if target is not None:
                lines.append(f"  Cible = {target:.6f}")

            if non_normal_info:
                lines.append(f"\nPercentiles ({distribution}) :")
                lines.append(f"  P(0.135%) = {non_normal_info['p00135']:.6f}")
                lines.append(f"  P(50%)    = {non_normal_info['p50']:.6f}")
                lines.append(f"  P(99.865%)= {non_normal_info['p99865']:.6f}")

            lines.append(f"\nIndice intra (dans) :")
            if "cp" in results:
                lines.append(f"  Cp  = {results['cp']:.2f}")
            if "cpk" in results:
                lines.append(f"  Cpk = {results['cpk']:.2f}")
            if "cpl" in results:
                lines.append(f"  Cpl = {results['cpl']:.2f}")
            if "cpu" in results:
                lines.append(f"  Cpu = {results['cpu']:.2f}")

            lines.append(f"\nIndice global :")
            if non_normal_info:
                if non_normal_info["pp"] is not None:
                    lines.append(f"  Pp  = {non_normal_info['pp']:.2f} (non-normal)")
                if non_normal_info["ppk"] is not None:
                    lines.append(f"  Ppk = {non_normal_info['ppk']:.2f} (non-normal)")
                if non_normal_info["ppl"] is not None:
                    lines.append(f"  Ppl = {non_normal_info['ppl']:.2f} (non-normal)")
                if non_normal_info["ppu"] is not None:
                    lines.append(f"  Ppu = {non_normal_info['ppu']:.2f} (non-normal)")
            else:
                if "pp" in results:
                    lines.append(f"  Pp  = {results['pp']:.2f}")
                if "ppk" in results:
                    lines.append(f"  Ppk = {results['ppk']:.2f}")
                if "ppl" in results:
                    lines.append(f"  Ppl = {results['ppl']:.2f}")
                if "ppu" in results:
                    lines.append(f"  Ppu = {results['ppu']:.2f}")

            if "cpm" in results:
                lines.append(f"\nCpm = {results['cpm']:.2f}")
            lines.append("=" * 60)
            self.cap_result_text.setText("\n".join(lines))

            self._plot_capability(analysis, results, distribution)
            self.status_bar.showMessage("Analyse de capabilité terminée")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors du calcul :\n{e}")

    def _plot_capability(self, analysis, results, distribution="Normal"):
        if self.cap_canvas is None or self.cap_canvas.fig is None:
            return
        self.cap_canvas.fig.clear()

        data, _ = self._get_data_from_sheet()
        mean = results["mean"]
        sigma_within = results["std_within"]
        sigma_overall = results["std_overall"]
        lsl = results.get("lsl")
        usl = results.get("usl")

        self.cap_canvas.fig.set_size_inches(14, 9)

        gs = self.cap_canvas.fig.add_gridspec(2, 2, left=0.06, right=0.96, top=0.94, bottom=0.08, wspace=0.25, hspace=0.30)

        ax1 = self.cap_canvas.fig.add_subplot(gs[0, 0])
        ax2 = self.cap_canvas.fig.add_subplot(gs[0, 1])
        ax3 = self.cap_canvas.fig.add_subplot(gs[1, 0])
        ax4 = self.cap_canvas.fig.add_subplot(gs[1, 1])

        if distribution == "Normal":
            fit_dist = "norm"
            is_normal = True
        else:
            dist_map = {
                "Log-Normale": ("lognorm", lambda d: scipy_stats.lognorm.fit(d, floc=0)),
                "Weibull (2P)": ("weibull_min", lambda d: scipy_stats.weibull_min.fit(d, floc=0)),
                "Exponentielle": ("expon", lambda d: scipy_stats.expon.fit(d)),
                "Gamma": ("gamma", lambda d: scipy_stats.gamma.fit(d, floc=0)),
                "Logistique": ("logistic", lambda d: scipy_stats.logistic.fit(d)),
                "Gumbel (max)": ("gumbel_r", lambda d: scipy_stats.gumbel_r.fit(d)),
                "Cauchy": ("cauchy", lambda d: scipy_stats.cauchy.fit(d)),
                "Rayleigh": ("rayleigh", lambda d: scipy_stats.rayleigh.fit(d, floc=0)),
                "Uniforme": ("uniform", lambda d: scipy_stats.uniform.fit(d)),
                "Student-t": ("t", lambda d: scipy_stats.t.fit(d)),
                "Laplace": ("laplace", lambda d: scipy_stats.laplace.fit(d)),
            }
            fit_info = dist_map.get(distribution, ("norm", lambda d: scipy_stats.norm.fit(d)))
            fit_dist = fit_info[0]
            fit_params = fit_info[1](data)
            is_normal = False
        fit_obj = getattr(scipy_stats, fit_dist)

        bins = max(10, int(np.sqrt(len(data))))

        p001 = np.percentile(data, 0.1)
        p999 = np.percentile(data, 99.9)
        span = p999 - p001
        x_min = p001 - 0.5 * span
        x_max = p999 + 0.5 * span
        x = np.linspace(x_min, x_max, 500)

        ax1.hist(data, bins=bins, density=True, alpha=0.6, color="lightblue", edgecolor="black")
        if is_normal:
            ax1.plot(x, scipy_stats.norm.pdf(x, mean, sigma_within), "r-", linewidth=2, label="Within")
            ax1.plot(x, scipy_stats.norm.pdf(x, mean, sigma_overall), "b--", linewidth=1.5, label="Overall")
            y_max = scipy_stats.norm.pdf(mean, mean, sigma_within)
        else:
            ax1.plot(x, fit_obj.pdf(x, *fit_params), "r-", linewidth=2, label=distribution)
            y_max = np.max(fit_obj.pdf(x, *fit_params))
        if lsl is not None:
            ax1.axvline(lsl, color="green", linestyle="-", linewidth=2)
        if usl is not None:
            ax1.axvline(usl, color="green", linestyle="-", linewidth=2)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, y_max * 1.15)
        ax1.legend(fontsize=7)
        cap_lines = []
        if results.get("cp") is not None:
            cap_lines.append(f"Cp={results['cp']:.2f}")
        if results.get("cpk") is not None:
            cap_lines.append(f"Cpk={results['cpk']:.2f}")
        if results.get("pp") is not None:
            cap_lines.append(f"Pp={results['pp']:.2f}")
        if results.get("ppk") is not None:
            cap_lines.append(f"Ppk={results['ppk']:.2f}")
        if results.get("ppk_non_normal") is not None:
            cap_lines.append(f"Ppk(NN)={results['ppk_non_normal']:.2f}")
        info_text = "\n".join(cap_lines)
        ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes, fontsize=8,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax1.set_title("Histogramme + Capabilité")

        if distribution == "Log-Normale":
            valid = data[data > 0]
            scipy_stats.probplot(np.log(valid), dist="norm", plot=ax2)
        else:
            scipy_stats.probplot(data, dist=fit_dist, plot=ax2)
        ax2.set_title("Graphique de probabilité")
        ax2.grid(True, alpha=0.3)

        ax3.plot(range(len(data)), data, "b-", linewidth=0.8, marker=".", markersize=3)
        ax3.axhline(mean, color="green", linestyle="-", linewidth=1.5)
        if lsl is not None:
            ax3.axhline(lsl, color="red", linestyle="--", linewidth=1)
        if usl is not None:
            ax3.axhline(usl, color="red", linestyle="--", linewidth=1)
        ax3.set_title("Run chart")
        ax3.set_xlabel("Observation")

        residuals = data - mean
        ax4.plot(range(len(data)), residuals, "b.", markersize=4)
        ax4.axhline(0, color="green", linestyle="-", linewidth=1.5)
        ax4.axhline(3 * sigma_within, color="red", linestyle="--", linewidth=1)
        ax4.axhline(-3 * sigma_within, color="red", linestyle="--", linewidth=1)
        ax4.set_title("Résidus")
        ax4.set_xlabel("Observation")

        try:
            self.cap_canvas.fig.tight_layout()
        except Exception:
            pass
        self.cap_canvas.draw()

    def _create_normality_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Normalité")
        left, left_layout, right, self.norm_canvas = self._setup_analysis_layout(tab)

        col_row = QHBoxLayout()
        self.norm_col_combo = QComboBox()
        self.norm_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.norm_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.norm_alpha = QDoubleSpinBox()
        self.norm_alpha.setRange(0.001, 0.5)
        self.norm_alpha.setSingleStep(0.005)
        self.norm_alpha.setValue(0.05)
        self._tip(self.norm_alpha, "Seuil de significativité α. Si p-value < α, on rejette l'hypothèse de normalité. 0.05 = risque 5%.")
        params_layout.addRow("Seuil α :", self.norm_alpha)

        tests_group = QGroupBox("Tests à effectuer")
        tests_layout = QVBoxLayout(tests_group)

        self.norm_tests = {}
        test_names = [
            ("Shapiro-Wilk", "Test le plus puissant pour N < 5000. Hypothèse H0: les données suivent une loi normale."),
            ("Anderson-Darling", "Plus sensible aux écarts dans les queues de distribution que Shapiro-Wilk."),
            ("D'Agostino-Pearson", "Combine l'asymétrie (skewness) et l'aplatissement (kurtosis). Bon pour grands échantillons."),
            ("Jarque-Bera", "Test asymptotique basé sur skewness et kurtosis. Adapté aux grands échantillons (N > 50)."),
            ("Ryan-Joiner", "Similaire à Shapiro-Wilk. Mesure la corrélation entre les données et les quantiles normaux."),
        ]
        for name, tooltip in test_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setToolTip(tooltip)
            self.norm_tests[name] = cb
            tests_layout.addWidget(cb)

        khi_row = QHBoxLayout()
        cb_chi = QCheckBox("Khi-deux (χ²)")
        cb_chi.setChecked(True)
        cb_chi.setToolTip("Test d'ajustement basé sur la comparaison des fréquences observées vs attendues dans des classes.")
        self.norm_tests["Khi-deux (χ²)"] = cb_chi
        khi_row.addWidget(cb_chi)
        khi_row.addWidget(QLabel("ddl :"))
        self.norm_chi2_df = QSpinBox()
        self.norm_chi2_df.setRange(1, 100)
        self.norm_chi2_df.setValue(10)
        self.norm_chi2_df.setMaximumWidth(60)
        self._tip(self.norm_chi2_df, "Degrés de liberté pour le test du Khi-deux. Par défaut = nombre de classes - 1 - paramètres estimés.")
        khi_row.addWidget(self.norm_chi2_df)
        khi_row.addStretch()
        tests_layout.addLayout(khi_row)

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
        data, col_name = self._get_combo_data(self.norm_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return

        try:
            alpha = self.norm_alpha.value()
            chi2_df = self.norm_chi2_df.value()
            tests = NormalityTests(data, alpha=alpha, chi2_df=chi2_df)
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

        col_row = QHBoxLayout()
        self.out_col_combo = QComboBox()
        self.out_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.out_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.out_alpha = QDoubleSpinBox()
        self.out_alpha.setRange(0.001, 0.5)
        self.out_alpha.setSingleStep(0.005)
        self.out_alpha.setValue(0.05)
        self._tip(self.out_alpha, "Seuil de détection. Plus petit = moins de valeurs détectées comme aberrantes.")
        params_layout.addRow("Seuil α :", self.out_alpha)

        tests_group = QGroupBox("Méthodes")
        tests_layout = QVBoxLayout(tests_group)

        self.out_tests = {}
        test_desc = [
            ("IQR", "Méthode de l'intervalle interquartile. Valeurs < Q1-1.5×IQR ou > Q3+1.5×IQR. Robuste, ne suppose pas de normalité."),
            ("Z-score", "Écart à la moyenne en unités d'écart-type. Seuil typique : |Z| > 3. Suppose la normalité."),
            ("Z-score modifié", "Utilise la médiane et le MAD (Median Absolute Deviation). Plus robuste aux outliers que Z-score classique."),
            ("Grubbs", "Test statistique pour détecter un outlier à la fois. Basé sur la distribution normale. Pour N ≥ 7."),
            ("Dixon", "Test Q de Dixon. Pour petits échantillons (N < 30). Compare l'écart entre valeurs extrêmes."),
            ("Chauvenet", "Critère probabiliste. Rejette une valeur si la probabilité de l'observer est < 1/(2N)."),
            ("Tukey (extrêmes)", "Variante stricte de IQR avec facteur 3× au lieu de 1.5×. Détecte seulement les outliers extrêmes."),
        ]
        for name, desc in test_desc:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setToolTip(desc)
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
        data, col_name = self._get_combo_data(self.out_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return

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

            self._plot_outliers(full_results, detector, data)
            self.status_bar.showMessage("Détection de valeurs aberrantes terminée")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de la détection :\n{e}")

    def _plot_outliers(self, full_results, detector, data):
        self.out_canvas.fig.clear()

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

    def _tip(self, widget, text):
        widget.setToolTip(text)

    def _copy_text(self, widget):
        QApplication.clipboard().setText(widget.toPlainText())

    def _add_copy_button(self, layout, text_widget):
        btn = QPushButton(" Copier")
        btn.clicked.connect(lambda: self._copy_text(text_widget))
        layout.addWidget(btn)

    def _refresh_combos(self, *combos):
        cols = [col_letter(i) for i in range(NUM_COLS)]
        has_data = False
        for i in range(NUM_COLS):
            if self.sheet.get_column_data(i) is not None:
                has_data = True
                break
        if not has_data:
            cols = ["(aucune donnée)"]
        for combo in combos:
            current = combo.currentText()
            combo.clear()
            combo.addItems(cols)
            if current in cols:
                combo.setCurrentText(current)

    def _get_combo_data(self, combo):
        letter = combo.currentText()
        if not letter or len(letter) != 1:
            return None, None
        idx = ord(letter) - 65
        return self.sheet.get_column_data(idx), letter

    def _get_all_numeric_columns(self):
        cols = {}
        for c in range(self.sheet.columnCount()):
            data = self.sheet.get_column_data(c)
            if data is not None and len(data) >= 2:
                cols[col_letter(c)] = data
        return cols

    # ===== CORRELATION =====
    def _create_correlation_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Corrélation")
        left, left_layout, right, self.corr_canvas = self._setup_analysis_layout(tab)

        col_row = QHBoxLayout()
        self.corr_col_combo = QComboBox()
        self.corr_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonnes :"))
        col_row.addWidget(self.corr_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.corr_method = QComboBox()
        self.corr_method.addItems(["Pearson", "Spearman", "Kendall"])
        self._tip(self.corr_method, "Pearson: linéaire, Spearman: monotone, Kendall: rangs concordants")
        params_layout.addRow("Méthode :", self.corr_method)

        self.corr_min_corr = QDoubleSpinBox()
        self.corr_min_corr.setRange(0, 1)
        self.corr_min_corr.setSingleStep(0.05)
        self.corr_min_corr.setValue(0.7)
        self._tip(self.corr_min_corr, "Seuil minimum pour mettre en évidence les corrélations fortes dans la matrice.")
        params_layout.addRow("Seuil corrélation forte :", self.corr_min_corr)

        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Calculer")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_correlation)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.corr_result_text = QTextEdit()
        self.corr_result_text.setReadOnly(True)
        self.corr_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.corr_result_text)
        self._add_copy_button(result_layout, self.corr_result_text)
        left_layout.addWidget(result_group)

    def _run_correlation(self):
        cols = self._get_all_numeric_columns()
        if len(cols) < 2:
            QMessageBox.warning(self, "Attention", "Au moins 2 colonnes avec données requises")
            return
        try:
            method = self.corr_method.currentText()
            min_corr = self.corr_min_corr.value()
            labels = list(cols.keys())
            data = np.column_stack([cols[k] for k in labels])
            analysis = CorrelationMatrix(data, labels=labels, method=method.lower())
            self.corr_result_text.setText(analysis.get_summary())
            self._plot_correlation(data, labels, method, min_corr)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_correlation(self, data, labels, method, min_corr=0.7):
        self.corr_canvas.fig.clear()
        ax = self.corr_canvas.fig.add_subplot(111)
        corr = np.corrcoef(data, rowvar=False)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if abs(corr[i, j]) > min_corr else "black"
                weight = "bold" if abs(corr[i, j]) > min_corr else "normal"
                text = ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                               color=color, fontsize=9, fontweight=weight)
        self.corr_canvas.fig.colorbar(im, ax=ax, label="Corrélation")
        ax.set_title(f"Matrice de corrélation ({method})")
        self.corr_canvas.fig.tight_layout()
        self.corr_canvas.draw()

    # ===== REGRESSION =====
    def _create_regression_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Régression")
        left, left_layout, right, self.reg_canvas = self._setup_analysis_layout(tab)
        col_row = QHBoxLayout()
        self.reg_y_combo = QComboBox()
        self.reg_y_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Y (réponse) :"))
        col_row.addWidget(self.reg_y_combo)
        self.reg_x_combo = QComboBox()
        self.reg_x_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("X (prédicteur) :"))
        col_row.addWidget(self.reg_x_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)
        self._refresh_all_combos()
        params_group = QGroupBox("Options")
        params_layout = QFormLayout(params_group)
        self.reg_show_ci = QCheckBox("Intervalle de confiance")
        self.reg_show_ci.setChecked(True)
        params_layout.addRow(self.reg_show_ci)
        self.reg_confidence = QDoubleSpinBox()
        self.reg_confidence.setRange(80, 99.9)
        self.reg_confidence.setValue(95)
        self.reg_confidence.setSuffix(" %")
        self._tip(self.reg_confidence, "Niveau de confiance pour les intervalles")
        params_layout.addRow("Confiance :", self.reg_confidence)
        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Calculer")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_regression)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.reg_result_text = QTextEdit()
        self.reg_result_text.setReadOnly(True)
        self.reg_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.reg_result_text)
        self._add_copy_button(result_layout, self.reg_result_text)
        left_layout.addWidget(result_group)

    def _run_regression(self):
        y_data, y_name = self._get_combo_data(self.reg_y_combo)
        x_data, x_name = self._get_combo_data(self.reg_x_combo)
        if y_data is None or x_data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez Y et X")
            return
        try:
            min_len = min(len(y_data), len(x_data))
            y_data = y_data[:min_len]
            x_data = x_data[:min_len]
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
            confidence = self.reg_confidence.value() / 100.0
            lines = ["=" * 60, "RÉGRESSION LINÉAIRE", "=" * 60,
                     f"\nY = {y_name}, X = {x_name}", f"N = {min_len}",
                     f"\nÉquation : Y = {intercept:.6f} + {slope:.6f} * X",
                     f"\nR² = {r_value**2:.6f}", f"R = {r_value:.6f}",
                     f"p-value = {p_value:.6f}", f"Erreur standard = {std_err:.6f}",
                     f"\nIntervalle confiance : {confidence*100:.0f}%"]
            self.reg_result_text.setText("\n".join(lines))
            self._plot_regression(x_data, y_data, slope, intercept)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_regression(self, x, y, slope, intercept):
        self.reg_canvas.fig.clear()
        ax = self.reg_canvas.fig.add_subplot(211)
        ax.scatter(x, y, alpha=0.6, color="blue", label="Données")
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, intercept + slope * x_line, "r-", linewidth=2, label=f"Y = {intercept:.2f} + {slope:.2f}X")
        if self.reg_show_ci.isChecked():
            n = len(x)
            mean_x = np.mean(x)
            ss_x = np.sum((x - mean_x)**2)
            y_pred = intercept + slope * x_line
            mse = np.sum((y - (intercept + slope * x))**2) / (n - 2)
            t_val = scipy_stats.t.ppf(0.975, n - 2)
            se_pred = np.sqrt(mse * (1/n + (x_line - mean_x)**2 / ss_x))
            ci_upper = y_pred + t_val * se_pred
            ci_lower = y_pred - t_val * se_pred
            ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color="red", label="IC 95%")
        ax.legend()
        ax.set_title("Régression linéaire")
        ax2 = self.reg_canvas.fig.add_subplot(212)
        residuals = y - (intercept + slope * x)
        ax2.scatter(x, residuals, alpha=0.6, color="green")
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_title("Résidus")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Résidu")
        self.reg_canvas.fig.tight_layout()
        self.reg_canvas.draw()

    # ===== T-TEST =====
    def _create_ttest_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Test t")
        left, left_layout, right, self.tt_canvas = self._setup_analysis_layout(tab)
        col_row = QHBoxLayout()
        self.tt_col1_combo = QComboBox()
        self.tt_col1_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne 1 :"))
        col_row.addWidget(self.tt_col1_combo)
        self.tt_col2_combo = QComboBox()
        self.tt_col2_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne 2 :"))
        col_row.addWidget(self.tt_col2_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)
        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)
        self.tt_type = QComboBox()
        self.tt_type.addItems(["Deux échantillons", "Un échantillon"])
        self._tip(self.tt_type, "Un échantillon: compare la moyenne à une valeur. Deux échantillons: compare deux moyennes.")
        params_layout.addRow("Type :", self.tt_type)
        self.tt_mu = QDoubleSpinBox()
        self.tt_mu.setRange(-1e9, 1e9)
        self.tt_mu.setValue(0)
        self._tip(self.tt_mu, "Valeur hypothétique de la moyenne (H0: μ = valeur)")
        params_layout.addRow("μ hypothétique :", self.tt_mu)
        self.tt_alpha = QDoubleSpinBox()
        self.tt_alpha.setRange(0.001, 0.5)
        self.tt_alpha.setValue(0.05)
        params_layout.addRow("Seuil α :", self.tt_alpha)
        self.tt_equal_var = QCheckBox("Variances égales")
        self.tt_equal_var.setChecked(True)
        params_layout.addRow(self.tt_equal_var)
        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Calculer")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_ttest)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.tt_result_text = QTextEdit()
        self.tt_result_text.setReadOnly(True)
        self.tt_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.tt_result_text)
        self._add_copy_button(result_layout, self.tt_result_text)
        left_layout.addWidget(result_group)

    def _run_ttest(self):
        data1, name1 = self._get_combo_data(self.tt_col1_combo)
        if data1 is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return
        try:
            alpha = self.tt_alpha.value()
            if self.tt_type.currentText() == "Un échantillon":
                mu = self.tt_mu.value()
                t_stat, p_value = scipy_stats.ttest_1samp(data1, mu)
                lines = ["=" * 60, "TEST T - UN ÉCHANTILLON", "=" * 60,
                         f"\nColonne : {name1}", f"N = {len(data1)}",
                         f"Moyenne = {np.mean(data1):.6f}", f"μ hypothétique = {mu}",
                         f"\nt = {t_stat:.6f}", f"p-value = {p_value:.6f}",
                         f"Seuil α = {alpha}",
                         f"\nH0 rejetée = {'OUI' if p_value < alpha else 'NON'}"]
            else:
                data2, name2 = self._get_combo_data(self.tt_col2_combo)
                if data2 is None:
                    QMessageBox.warning(self, "Attention", "Sélectionnez une 2ème colonne")
                    return
                equal_var = self.tt_equal_var.isChecked()
                t_stat, p_value = scipy_stats.ttest_ind(data1, data2, equal_var=equal_var)
                lines = ["=" * 60, "TEST T - DEUX ÉCHANTILLONS", "=" * 60,
                         f"\nColonne 1 : {name1} (N={len(data1)}, moy={np.mean(data1):.4f})",
                         f"Colonne 2 : {name2} (N={len(data2)}, moy={np.mean(data2):.4f})",
                         f"\nt = {t_stat:.6f}", f"p-value = {p_value:.6f}",
                         f"Variances égales = {'Oui' if equal_var else 'Non'}",
                         f"Seuil α = {alpha}",
                         f"\nH0 rejetée = {'OUI' if p_value < alpha else 'NON'}"]
            self.tt_result_text.setText("\n".join(lines))
            self._plot_ttest(data1, data2 if self.tt_type.currentText() == "Deux échantillons" else None)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_ttest(self, data1, data2):
        self.tt_canvas.fig.clear()
        ax = self.tt_canvas.fig.add_subplot(111)
        if data2 is not None:
            ax.boxplot([data1, data2], labels=[self.tt_col1_combo.currentText(), self.tt_col2_combo.currentText()],
                      patch_artist=True)
        else:
            ax.boxplot([data1], labels=[self.tt_col1_combo.currentText()], patch_artist=True)
            ax.axhline(self.tt_mu.value(), color="red", linestyle="--", label=f"μ={self.tt_mu.value()}")
            ax.legend()
        ax.set_title("Comparaison")
        self.tt_canvas.fig.tight_layout()
        self.tt_canvas.draw()

    # ===== ANOVA =====
    def _create_anova_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "ANOVA")
        left, left_layout, right, self.anova_canvas = self._setup_analysis_layout(tab)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.anova_alpha = QDoubleSpinBox()
        self.anova_alpha.setRange(0.001, 0.5)
        self.anova_alpha.setValue(0.05)
        self._tip(self.anova_alpha, "Seuil α pour le test ANOVA. Probabilité de rejeter H0 si elle est vraie.")
        params_layout.addRow("Seuil α :", self.anova_alpha)

        self.anova_ngroups = QSpinBox()
        self.anova_ngroups.setRange(2, 20)
        self.anova_ngroups.setValue(3)
        self._tip(self.anova_ngroups, "Nombre de groupes à comparer. Minimum 2, maximum 20.")
        params_layout.addRow("Nombre de groupes :", self.anova_ngroups)

        left_layout.addWidget(params_group)

        self.anova_col_combos = []
        self.anova_combo_container = QWidget()
        self.anova_combo_layout = QVBoxLayout(self.anova_combo_container)
        self.anova_combo_layout.setContentsMargins(0, 0, 0, 0)
        self._update_anova_combos()
        left_layout.addWidget(self.anova_combo_container)

        self.anova_ngroups.valueChanged.connect(self._update_anova_combos)

        calc_btn = QPushButton(" Calculer ANOVA")
        calc_btn.setMinimumHeight(40)
        calc_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        calc_btn.clicked.connect(self._run_anova)
        left_layout.addWidget(calc_btn)

        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.anova_result_text = QTextEdit()
        self.anova_result_text.setReadOnly(True)
        self.anova_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.anova_result_text)
        self._add_copy_button(result_layout, self.anova_result_text)
        left_layout.addWidget(result_group)

    def _update_anova_combos(self):
        current_selections = []
        if hasattr(self, 'anova_col_combos') and self.anova_col_combos:
            for combo in self.anova_col_combos:
                current_selections.append(combo.currentText())

        while self.anova_combo_layout.count():
            item = self.anova_combo_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.anova_col_combos = []

        cols = []
        for i in range(self.sheet.columnCount()):
            data = self.sheet.get_column_data(i)
            if data is not None and len(data) > 0:
                cols.append(col_letter(i))
        if not cols:
            cols = ["(aucune donnée)"]

        n = self.anova_ngroups.value()
        for i in range(n):
            row = QHBoxLayout()
            label = QLabel(f"Groupe {i+1} :")
            label.setMinimumWidth(60)
            row.addWidget(label)
            combo = QComboBox()
            combo.setMinimumWidth(80)
            combo.addItems(cols)
            if i < len(current_selections) and current_selections[i] in cols:
                combo.setCurrentText(current_selections[i])
            row.addWidget(combo)
            row.addStretch()
            self.anova_combo_layout.addLayout(row)
            self.anova_col_combos.append(combo)

    def _run_anova(self):
        groups = {}
        for i, combo in enumerate(self.anova_col_combos):
            letter = combo.currentText()
            if not letter or len(letter) != 1:
                continue
            idx = ord(letter) - 65
            data = self.sheet.get_column_data(idx)
            if data is not None and len(data) > 0:
                groups[f"G{i+1} ({letter})"] = data

        if len(groups) < 2:
            QMessageBox.warning(self, "Attention", "Au moins 2 groupes avec données requis pour ANOVA")
            return

        try:
            alpha = self.anova_alpha.value()
            group_data = list(groups.values())
            group_labels = list(groups.keys())
            f_stat, p_value = scipy_stats.f_oneway(*group_data)

            ss_between = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(group_data)))**2 for g in group_data)
            ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in group_data)
            ss_total = ss_between + ss_within

            k = len(group_data)
            n_total = sum(len(g) for g in group_data)
            df_between = k - 1
            df_within = n_total - k
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within

            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            lines = ["=" * 60, "ANOVA - Analyse de variance", "=" * 60,
                     f"\nNombre de groupes : {k}",
                     f"N total = {n_total}",
                     f"Seuil α = {alpha}",
                     f"\nSource      |   SS      |  df  |   MS      |    F"]
            lines.append("-" * 55)
            lines.append(f"Entre      | {ss_between:10.4f} | {df_between:4d} | {ms_between:10.4f} | {f_stat:.4f}")
            lines.append(f"Intra      | {ss_within:10.4f} | {df_within:4d} | {ms_within:10.4f} |")
            lines.append(f"Total      | {ss_total:10.4f} | {n_total-1:4d} |           |")
            lines.append(f"\np-value = {p_value:.6f}")
            lines.append(f"η² (eta-squared) = {eta_sq:.4f}")
            lines.append(f"\nH0 rejetée = {'OUI' if p_value < alpha else 'NON'}")
            if p_value < alpha:
                lines.append("\n→ Au moins un groupe diffère significativement")
            lines.append("\nStatistiques par groupe :")
            for name, data in groups.items():
                lines.append(f"  {name}: N={len(data)}, moy={np.mean(data):.4f}, std={np.std(data, ddof=1):.4f}")
            self.anova_result_text.setText("\n".join(lines))
            self._plot_anova(group_data, group_labels)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_anova(self, group_data, group_labels):
        self.anova_canvas.fig.clear()
        ax = self.anova_canvas.fig.add_subplot(111)
        bp = ax.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(group_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title("ANOVA - Comparaison des groupes")
        ax.set_ylabel("Valeur")
        ax.tick_params(axis='x', rotation=15)
        try:
            self.anova_canvas.fig.tight_layout()
        except Exception:
            pass
        self.anova_canvas.draw()

    # ===== BOXPLOTS =====
    def _create_boxplot_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Boxplots")
        left, left_layout, right, self.boxplot_canvas = self._setup_analysis_layout(tab)

        self.boxplot_ngroups = QSpinBox()
        self.boxplot_ngroups.setRange(1, 20)
        self.boxplot_ngroups.setValue(3)
        self._tip(self.boxplot_ngroups, "Nombre de séries de données à comparer. Minimum 1, maximum 20.")
        params_layout.addRow("Nombre de séries :", self.boxplot_ngroups)

        left_layout.addWidget(params_group)

        self.boxplot_col_combos = []
        self.boxplot_combo_container = QWidget()
        self.boxplot_combo_layout = QVBoxLayout(self.boxplot_combo_container)
        self.boxplot_combo_layout.setContentsMargins(0, 0, 0, 0)
        self._update_boxplot_combos()
        left_layout.addWidget(self.boxplot_combo_container)

        params_group = QGroupBox("Options")
        params_layout = QFormLayout(params_group)
        self.boxplot_notched = QCheckBox("Boîtes à encoches")
        self.boxplot_notched.setChecked(False)
        self._tip(self.boxplot_notched, "Les encoches montrent l'IC de la médiane. Si les encoches ne se chevauchent pas, les médianes sont significativement différentes.")
        params_layout.addRow(self.boxplot_notched)
        self.boxplot_show_outliers = QCheckBox("Afficher valeurs aberrantes")
        self.boxplot_show_outliers.setChecked(True)
        self._tip(self.boxplot_show_outliers, "Affiche les points au-delà de 1.5×IQR comme des marqueurs individuels.")
        params_layout.addRow(self.boxplot_show_outliers)
        left_layout.addWidget(params_group)

        self.boxplot_ngroups.valueChanged.connect(self._update_boxplot_combos)

        calc_btn = QPushButton(" Tracer")
        calc_btn.setMinimumHeight(40)
        calc_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        calc_btn.clicked.connect(self._run_boxplot)
        left_layout.addWidget(calc_btn)

        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.boxplot_result_text = QTextEdit()
        self.boxplot_result_text.setReadOnly(True)
        self.boxplot_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.boxplot_result_text)
        self._add_copy_button(result_layout, self.boxplot_result_text)
        left_layout.addWidget(result_group)

    def _update_boxplot_combos(self):
        current_selections = []
        if hasattr(self, 'boxplot_col_combos') and self.boxplot_col_combos:
            for combo in self.boxplot_col_combos:
                current_selections.append(combo.currentText())

        while self.boxplot_combo_layout.count():
            item = self.boxplot_combo_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.boxplot_col_combos = []

        cols = []
        for i in range(self.sheet.columnCount()):
            data = self.sheet.get_column_data(i)
            if data is not None and len(data) > 0:
                cols.append(col_letter(i))
        if not cols:
            cols = ["(aucune donnée)"]

        n = self.boxplot_ngroups.value()
        for i in range(n):
            row = QHBoxLayout()
            label = QLabel(f"Série {i+1} :")
            label.setMinimumWidth(60)
            row.addWidget(label)
            combo = QComboBox()
            combo.setMinimumWidth(80)
            combo.addItems(cols)
            if i < len(current_selections) and current_selections[i] in cols:
                combo.setCurrentText(current_selections[i])
            row.addWidget(combo)
            row.addStretch()
            self.boxplot_combo_layout.addLayout(row)
            self.boxplot_col_combos.append(combo)

    def _run_boxplot(self):
        groups = {}
        for i, combo in enumerate(self.boxplot_col_combos):
            letter = combo.currentText()
            if not letter or len(letter) != 1:
                continue
            idx = ord(letter) - 65
            data = self.sheet.get_column_data(idx)
            if data is not None and len(data) > 0:
                groups[f"S{i+1} ({letter})"] = data

        if not groups:
            QMessageBox.warning(self, "Attention", "Aucune série avec données")
            return

        try:
            lines = ["=" * 60, "BOXPLOTS", "=" * 60]
            for name, data in groups.items():
                q1, q2, q3 = np.percentile(data, [25, 50, 75])
                iqr = q3 - q1
                lines.append(f"\n{name}: N={len(data)}, Q1={q1:.4f}, Méd={q2:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
            self.boxplot_result_text.setText("\n".join(lines))
            self._plot_boxplots(groups)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_boxplots(self, groups):
        self.boxplot_canvas.fig.clear()
        ax = self.boxplot_canvas.fig.add_subplot(111)
        data = list(groups.values())
        labels = list(groups.keys())
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                       notch=self.boxplot_notched.isChecked(),
                       showfliers=self.boxplot_show_outliers.isChecked())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title("Boxplots comparatifs")
        ax.set_ylabel("Valeur")
        ax.tick_params(axis='x', rotation=15)
        try:
            self.boxplot_canvas.fig.tight_layout()
        except Exception:
            pass
        self.boxplot_canvas.draw()

    # ===== CONTROL CHARTS =====
    def _create_control_charts_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Cartes de contrôle")
        left, left_layout, right, self.cc_canvas = self._setup_analysis_layout(tab)
        col_row = QHBoxLayout()
        self.cc_col_combo = QComboBox()
        self.cc_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.cc_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)
        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)
        self.cc_type = QComboBox()
        self.cc_type.addItems(["X̄-R", "X̄-S", "I-MR", "EWMA", "CUSUM", "P", "U", "C"])
        self._tip(self.cc_type, "X̄-R/X̄-S: sous-groupes. I-MR: individuelles. EWMA: moyenne mobile pondérée. CUSUM: cumulé. P/U/C: comptage.")
        params_layout.addRow("Type :", self.cc_type)
        self.cc_subgroup = QSpinBox()
        self.cc_subgroup.setRange(2, 25)
        self.cc_subgroup.setValue(5)
        self._tip(self.cc_subgroup, "Taille du sous-groupe pour X̄-R et X̄-S")
        params_layout.addRow("Sous-groupe :", self.cc_subgroup)

        self.cc_lambda = QDoubleSpinBox()
        self.cc_lambda.setRange(0.01, 1.0)
        self.cc_lambda.setSingleStep(0.05)
        self.cc_lambda.setValue(0.2)
        self._tip(self.cc_lambda, "Facteur de pondération pour EWMA (0.1-0.3 recommandé). Plus petit = détection lente mais sensible aux petits dérifts.")
        params_layout.addRow("Lambda (EWMA) :", self.cc_lambda)

        self.cc_k = QDoubleSpinBox()
        self.cc_k.setRange(0.1, 2.0)
        self.cc_k.setSingleStep(0.1)
        self.cc_k.setValue(0.5)
        self._tip(self.cc_k, "Valeur de référence pour CUSUM (k). 0.5σ pour détecter un dérift de 1σ.")
        params_layout.addRow("k (CUSUM) :", self.cc_k)

        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Tracer")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_control_charts)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.cc_result_text = QTextEdit()
        self.cc_result_text.setReadOnly(True)
        self.cc_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.cc_result_text)
        self._add_copy_button(result_layout, self.cc_result_text)
        left_layout.addWidget(result_group)

    def _run_control_charts(self):
        data, col_name = self._get_combo_data(self.cc_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return
        try:
            cc_type = self.cc_type.currentText()
            subgroup = self.cc_subgroup.value()
            lines = ["=" * 60, f"CARTE DE CONTRÔLE : {cc_type}", "=" * 60,
                     f"\nColonne : {col_name}", f"N = {len(data)}"]
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val
            lines.append(f"\nMoyenne = {mean_val:.6f}")

            if cc_type == "EWMA":
                lam = self.cc_lambda.value()
                z = np.zeros(len(data))
                z[0] = mean_val
                for i in range(1, len(data)):
                    z[i] = lam * data[i] + (1 - lam) * z[i-1]
                sigma_z = std_val * np.sqrt(lam / (2 - lam) * (1 - (1 - lam)**(2 * np.arange(1, len(data) + 1))))
                ucl_ewma = mean_val + 3 * sigma_z
                lcl_ewma = mean_val - 3 * sigma_z
                lines.append(f"Lambda = {lam:.3f}")
                lines.append(f"UCL (asymptotique) = {mean_val + 3 * std_val * np.sqrt(lam / (2 - lam)):.6f}")
                lines.append(f"LCL (asymptotique) = {mean_val - 3 * std_val * np.sqrt(lam / (2 - lam)):.6f}")
                out_of_control = np.sum((z > ucl_ewma) | (z < lcl_ewma))
            elif cc_type == "CUSUM":
                k = self.cc_k.value() * std_val
                h = 5 * std_val
                s_pos = np.zeros(len(data))
                s_neg = np.zeros(len(data))
                for i in range(1, len(data)):
                    s_pos[i] = max(0, s_pos[i-1] + data[i] - mean_val - k)
                    s_neg[i] = max(0, s_neg[i-1] + mean_val - data[i] - k)
                lines.append(f"k = {k:.6f}, h = {h:.6f}")
                signals_pos = np.where(s_pos > h)[0]
                signals_neg = np.where(s_neg > h)[0]
                out_of_control = len(signals_pos) + len(signals_neg)
            else:
                lines.append(f"UCL = {ucl:.6f}")
                lines.append(f"LCL = {lcl:.6f}")
                out_of_control = np.sum((data > ucl) | (data < lcl))

            lines.append(f"\nPoints hors contrôle : {out_of_control}/{len(data)}")
            self.cc_result_text.setText("\n".join(lines))
            self._plot_control_charts(data, cc_type, subgroup)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_control_charts(self, data, cc_type, subgroup):
        self.cc_canvas.fig.clear()
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        if cc_type == "EWMA":
            lam = self.cc_lambda.value()
            z = np.zeros(len(data))
            z[0] = mean_val
            for i in range(1, len(data)):
                z[i] = lam * data[i] + (1 - lam) * z[i-1]
            sigma_z = std_val * np.sqrt(lam / (2 - lam) * (1 - (1 - lam)**(2 * np.arange(1, len(data) + 1))))
            ucl_ewma = mean_val + 3 * sigma_z
            lcl_ewma = mean_val - 3 * sigma_z

            ax1 = self.cc_canvas.fig.add_subplot(211)
            ax2 = self.cc_canvas.fig.add_subplot(212)
            ax1.plot(range(len(data)), data, "bo-", markersize=3, alpha=0.5, label="Données")
            ax1.axhline(mean_val, color="green", linestyle="--", label="CL")
            ax1.set_title("Données individuelles")
            ax1.legend(fontsize=7)
            ax2.plot(range(1, len(data)+1), z, "ro-", markersize=3, label="EWMA")
            ax2.fill_between(range(1, len(data)+1), lcl_ewma, ucl_ewma, alpha=0.15, color="red", label="Limites")
            ax2.plot(range(1, len(data)+1), ucl_ewma, "r--", linewidth=1, label="UCL")
            ax2.plot(range(1, len(data)+1), lcl_ewma, "r--", linewidth=1, label="LCL")
            ax2.axhline(mean_val, color="green", linestyle="--", linewidth=1)
            ax2.set_title(f"Carte EWMA (λ={lam:.2f})")
            ax2.legend(fontsize=7)

        elif cc_type == "CUSUM":
            k = self.cc_k.value() * std_val
            h = 5 * std_val
            s_pos = np.zeros(len(data))
            s_neg = np.zeros(len(data))
            for i in range(1, len(data)):
                s_pos[i] = max(0, s_pos[i-1] + data[i] - mean_val - k)
                s_neg[i] = max(0, s_neg[i-1] + mean_val - data[i] - k)

            ax1 = self.cc_canvas.fig.add_subplot(211)
            ax2 = self.cc_canvas.fig.add_subplot(212)
            ax1.plot(range(len(data)), data, "bo-", markersize=3, alpha=0.5)
            ax1.axhline(mean_val, color="green", linestyle="--", label="CL")
            ax1.set_title("Données individuelles")
            ax1.legend(fontsize=7)
            ax2.plot(range(len(data)), s_pos, "r-", markersize=2, label="CUSUM+")
            ax2.plot(range(len(data)), s_neg, "b-", markersize=2, label="CUSUM-")
            ax2.axhline(h, color="red", linestyle="--", linewidth=1, label="h")
            ax2.axhline(-h, color="red", linestyle="--", linewidth=1)
            ax2.axhline(0, color="green", linestyle="--", linewidth=1)
            ax2.set_title(f"Carte CUSUM (k={k:.3f}, h={h:.3f})")
            ax2.legend(fontsize=7)

        elif cc_type in ["X̄-R", "X̄-S"]:
            ax1 = self.cc_canvas.fig.add_subplot(211)
            ax2 = self.cc_canvas.fig.add_subplot(212)
            n_subgroups = len(data) // subgroup
            subgroups = data[:n_subgroups * subgroup].reshape(n_subgroups, subgroup)
            means = np.mean(subgroups, axis=1)
            ranges = np.ptp(subgroups, axis=1)
            ax1.plot(range(len(means)), means, "bo-", markersize=4)
            ax1.axhline(np.mean(means), color="green", linestyle="--", label="CL")
            ax1.axhline(np.mean(means) + 3 * np.std(means, ddof=1), color="red", linestyle="-.", label="UCL")
            ax1.axhline(np.mean(means) - 3 * np.std(means, ddof=1), color="red", linestyle="-.", label="LCL")
            ax1.set_title(f"Carte {cc_type.split('-')[0]}")
            ax1.legend(fontsize=7)
            ax2.plot(range(len(ranges)), ranges, "bo-", markersize=4)
            ax2.axhline(np.mean(ranges), color="green", linestyle="--")
            ax2.set_title("Carte R")
        else:
            ax1 = self.cc_canvas.fig.add_subplot(211)
            ax2 = self.cc_canvas.fig.add_subplot(212)
            ax1.plot(range(len(data)), data, "bo-", markersize=3)
            ax1.axhline(mean_val, color="green", linestyle="--", label="CL")
            ax1.axhline(ucl, color="red", linestyle="-.", label="UCL")
            ax1.axhline(lcl, color="red", linestyle="-.", label="LCL")
            ax1.set_title(f"Carte {cc_type}")
            ax1.legend(fontsize=7)
            ax2.hist(data, bins=20, density=True, alpha=0.6, color="lightblue")
            ax2.set_title("Distribution")

        self.cc_canvas.fig.tight_layout()
        self.cc_canvas.draw()

    # ===== PROBABILITY PLOTS =====
    def _create_probplot_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "Graphiques probabilité")
        left, left_layout, right, self.prob_canvas = self._setup_analysis_layout(tab)
        col_row = QHBoxLayout()
        self.prob_col_combo = QComboBox()
        self.prob_col_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Colonne :"))
        col_row.addWidget(self.prob_col_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)
        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)
        self.prob_dist = QComboBox()
        self.prob_dist.addItems([
            "Normale", "Log-Normale", "Weibull (2P)", "Weibull (3P)",
            "Exponentielle", "Gamma", "Logistique", "Gumbel (max)",
            "Gumbel (min)", "Cauchy", "Rayleigh", "Uniforme",
            "Student-t", "Laplace", "Beta", "Pareto",
        ])
        self._tip(self.prob_dist, "Distribution théorique pour le graphique de probabilité")
        params_layout.addRow("Distribution :", self.prob_dist)
        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Tracer")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_probplot)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.prob_result_text = QTextEdit()
        self.prob_result_text.setReadOnly(True)
        self.prob_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.prob_result_text)
        self._add_copy_button(result_layout, self.prob_result_text)
        left_layout.addWidget(result_group)

    def _run_probplot(self):
        data, col_name = self._get_combo_data(self.prob_col_combo)
        if data is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez une colonne")
            return
        try:
            dist_map = {
                "Normale": "normal", "Log-Normale": "lognormal",
                "Weibull (2P)": "weibull_min", "Weibull (3P)": "weibull_min",
                "Exponentielle": "exponential", "Gamma": "gamma",
                "Logistique": "logistic", "Gumbel (max)": "gumbel_max",
                "Gumbel (min)": "gumbel_min", "Cauchy": "cauchy",
                "Rayleigh": "rayleigh", "Uniforme": "uniform",
                "Student-t": "t", "Laplace": "laplace",
                "Beta": "beta", "Pareto": "pareto",
            }
            dist = self.prob_dist.currentText()
            analysis = ProbabilityPlot(data, distribution=dist_map.get(dist, "normal"))
            self.prob_result_text.setText(analysis.get_summary())
            self._plot_probplot(data, dist_map.get(dist, "normal"))
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_probplot(self, data, dist):
        self.prob_canvas.fig.clear()
        ax = self.prob_canvas.fig.add_subplot(111)

        dist_map = {
            "normal": "norm", "lognormal": "lognorm",
            "weibull_min": "weibull_min", "exponential": "expon",
            "gamma": "gamma", "logistic": "logistic",
            "gumbel_max": "gumbel_r", "gumbel_min": "gumbel_l",
            "cauchy": "cauchy", "rayleigh": "rayleigh",
            "uniform": "uniform", "t": "t",
            "laplace": "laplace", "beta": "beta",
            "pareto": "pareto",
        }
        d = dist_map.get(dist, "norm")

        if dist == "lognormal":
            valid = data[data > 0]
            if len(valid) > 0:
                scipy_stats.probplot(np.log(valid), dist="norm", plot=ax)
            else:
                ax.text(0.5, 0.5, "Données invalides", ha='center', va='center', transform=ax.transAxes)
        elif dist == "beta":
            d_min, d_max = np.min(data), np.max(data)
            rng = d_max - d_min
            if rng == 0:
                ax.text(0.5, 0.5, "Données constantes", ha='center', va='center', transform=ax.transAxes)
            else:
                eps = rng * 0.001
                scaled = (data - d_min + eps) / (rng + 2 * eps)
                a, b = scipy_stats.beta.fit(scaled, floc=0, fscale=1)
                n = len(data)
                sorted_data = np.sort(data)
                percentiles = np.arange(1, n + 1) / (n + 1)
                theoretical_scaled = scipy_stats.beta.ppf(percentiles, a, b, 0, 1)
                theoretical_orig = d_min + theoretical_scaled * rng
                ax.plot(theoretical_orig, sorted_data, 'o', markersize=3, alpha=0.7)
                min_val = min(theoretical_orig.min(), sorted_data.min())
                max_val = max(theoretical_orig.max(), sorted_data.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5)
        else:
            dist_obj = getattr(scipy_stats, d)
            try:
                params = dist_obj.fit(data)
                n = len(data)
                sorted_data = np.sort(data)
                percentiles = np.arange(1, n + 1) / (n + 1)
                theoretical = dist_obj.ppf(percentiles, *params)
                ax.plot(theoretical, sorted_data, 'o', markersize=3, alpha=0.7)
                min_val = min(theoretical.min(), sorted_data.min())
                max_val = max(theoretical.max(), sorted_data.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5)
            except Exception:
                scipy_stats.probplot(data, dist=d, plot=ax)

        ax.set_title(f"Graphique de probabilité - {dist}")
        ax.set_xlabel("Valeurs théoriques")
        ax.set_ylabel("Valeurs observées")
        ax.grid(True, alpha=0.3)
        self.prob_canvas.fig.tight_layout()
        self.prob_canvas.draw()

    # ===== MSA =====
    def _create_msa_tab(self):
        tab = QWidget()
        self.tabs.addTab(tab, "MSA / Gage R&R")
        left, left_layout, right, self.msa_canvas = self._setup_analysis_layout(tab)
        info = QLabel("Format : 3 colonnes — Pièces, Opérateurs, Mesures")
        info.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        left_layout.addWidget(info)

        col_row = QHBoxLayout()
        self.msa_part_combo = QComboBox()
        self.msa_part_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Pièces :"))
        col_row.addWidget(self.msa_part_combo)
        self.msa_op_combo = QComboBox()
        self.msa_op_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Opérateurs :"))
        col_row.addWidget(self.msa_op_combo)
        self.msa_meas_combo = QComboBox()
        self.msa_meas_combo.setMinimumWidth(80)
        col_row.addWidget(QLabel("Mesures :"))
        col_row.addWidget(self.msa_meas_combo)
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_all_combos)
        col_row.addWidget(refresh_btn)
        left_layout.addLayout(col_row)

        params_group = QGroupBox("Paramètres")
        params_layout = QFormLayout(params_group)

        self.msa_n_parts = QSpinBox()
        self.msa_n_parts.setRange(2, 50)
        self.msa_n_parts.setValue(10)
        self._tip(self.msa_n_parts, "Nombre de pièces dans l'étude. Typiquement 10.")
        params_layout.addRow("Nombre de pièces :", self.msa_n_parts)

        self.msa_n_operators = QSpinBox()
        self.msa_n_operators.setRange(2, 10)
        self.msa_n_operators.setValue(3)
        self._tip(self.msa_n_operators, "Nombre d'opérateurs. Typiquement 2-3.")
        params_layout.addRow("Nombre d'opérateurs :", self.msa_n_operators)

        self.msa_n_trials = QSpinBox()
        self.msa_n_trials.setRange(1, 10)
        self.msa_n_trials.setValue(3)
        self._tip(self.msa_n_trials, "Nombre de répétitions par opérateur/pièce. Typiquement 2-3.")
        params_layout.addRow("Nombre de répétitions :", self.msa_n_trials)

        self.msa_tolerance = QDoubleSpinBox()
        self.msa_tolerance.setRange(0.001, 1e9)
        self.msa_tolerance.setValue(1.0)
        self._tip(self.msa_tolerance, "Tolérance du processus (USL - LSL) pour le calcul P/T")
        params_layout.addRow("Tolérance :", self.msa_tolerance)

        left_layout.addWidget(params_group)
        calc_btn = QPushButton(" Calculer Gage R&R")
        calc_btn.setMinimumHeight(40)
        calc_btn.clicked.connect(self._run_msa)
        left_layout.addWidget(calc_btn)
        result_group = QGroupBox("Résultats")
        result_layout = QVBoxLayout(result_group)
        self.msa_result_text = QTextEdit()
        self.msa_result_text.setReadOnly(True)
        self.msa_result_text.setFont(QFont("Courier", 10))
        result_layout.addWidget(self.msa_result_text)
        self._add_copy_button(result_layout, self.msa_result_text)
        left_layout.addWidget(result_group)

    def _run_msa(self):
        parts_col, _ = self._get_combo_data(self.msa_part_combo)
        ops_col, _ = self._get_combo_data(self.msa_op_combo)
        meas_col, _ = self._get_combo_data(self.msa_meas_combo)
        if parts_col is None or ops_col is None or meas_col is None:
            QMessageBox.warning(self, "Attention", "Sélectionnez Pièces, Opérateurs et Mesures")
            return

        try:
            n_parts = self.msa_n_parts.value()
            n_operators = self.msa_n_operators.value()
            n_trials = self.msa_n_trials.value()
            tolerance = self.msa_tolerance.value()

            min_len = min(len(parts_col), len(ops_col), len(meas_col))
            parts_col = np.array(parts_col[:min_len])
            ops_col = np.array(ops_col[:min_len])
            meas_col = np.array(meas_col[:min_len])

            valid = ~np.isnan(parts_col) & ~np.isnan(ops_col) & ~np.isnan(meas_col)
            parts_col = parts_col[valid]
            ops_col = ops_col[valid]
            meas_col = meas_col[valid]

            if len(parts_col) == 0:
                QMessageBox.warning(self, "Attention", "Aucune donnée valide (sans NaN)")
                return

            unique_parts = np.unique(parts_col)
            unique_ops = np.unique(ops_col)

            data_3d = np.zeros((n_operators, n_parts, n_trials))
            for i, op in enumerate(sorted(unique_ops)):
                if i >= n_operators:
                    break
                for j, part in enumerate(sorted(unique_parts)):
                    if j >= n_parts:
                        break
                    mask = (parts_col == part) & (ops_col == op)
                    vals = meas_col[mask][:n_trials]
                    data_3d[i, j, :len(vals)] = vals

            study = GageRRStudy(data_3d, n_operators=n_operators, n_parts=n_parts,
                                n_trials=n_trials, tolerance=tolerance)
            self.msa_result_text.setText(study.get_summary())
            results = study.get_results()
            self._plot_msa(results)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _plot_msa(self, results):
        self.msa_canvas.fig.clear()
        ax1 = self.msa_canvas.fig.add_subplot(221)
        ax2 = self.msa_canvas.fig.add_subplot(222)
        ax3 = self.msa_canvas.fig.add_subplot(223)
        ax4 = self.msa_canvas.fig.add_subplot(224)

        labels = ["Répétabilité", "Reproductibilité", "Pièces", "Total"]
        values = [
            results.get('pct_contribution_repeatability', 0),
            results.get('pct_contribution_reproducibility', 0),
            results.get('pct_contribution_part', 0),
            results.get('pct_contribution_gage_rr', 0),
        ]
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        ax1.bar(labels, values, color=colors_bar)
        ax1.set_ylabel("% Contribution")
        ax1.set_title("Composantes de variance")
        ax1.tick_params(axis='x', rotation=15)

        op_means = results.get('op_means', [])
        if op_means:
            ax2.bar(range(len(op_means)), op_means, color='#3498db')
            ax2.set_xticks(range(len(op_means)))
            ax2.set_xticklabels([f"Op{i+1}" for i in range(len(op_means))])
            ax2.set_title("Moyennes par opérateur")

        part_means = results.get('part_means', [])
        if part_means:
            ax3.plot(range(len(part_means)), part_means, 'bo-', markersize=4)
            ax3.set_xticks(range(len(part_means)))
            ax3.set_xticklabels([f"P{i+1}" for i in range(len(part_means))], rotation=45, ha='right')
            ax3.set_title("Moyennes par pièce")

        rr_pct = results.get('pct_contribution_gage_rr', 0)
        part_pct = results.get('pct_contribution_part', 0)
        ax4.pie([rr_pct, max(0, 100 - rr_pct)], labels=['Gage R&R', 'Autre'],
                autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'])
        ax4.set_title("Répartition % contribution")

        self.msa_canvas.fig.tight_layout()
        self.msa_canvas.draw()

    def _export_results(self):
        texts = []
        for title, widget in [("CAPABILITÉ", self.cap_result_text), ("NORMALITÉ", self.norm_result_text),
                               ("VALEURS ABERRANTES", self.out_result_text), ("CARTES DE CONTRÔLE", self.cc_result_text),
                               ("GRAPHIQUES PROBABILITÉ", self.prob_result_text),
                               ("RÉGRESSION", self.reg_result_text), ("TEST T", self.tt_result_text),
                               ("ANOVA", self.anova_result_text), ("CORRÉLATION", self.corr_result_text),
                               ("BOXPLOTS", self.boxplot_result_text),
                               ("MSA / GAGE R&R", self.msa_result_text),
                               ("IDENTIFICATION DISTRIBUTION", self.dist_result_text)]:
            if widget.toPlainText().strip():
                texts.append((title, widget.toPlainText()))
        if not texts:
            QMessageBox.warning(self, "Attention", "Aucun résultat à exporter")
            return

        file_filters = (
            "Text (*.txt);;"
            "PDF (*.pdf);;"
            "ODT (*.odt);;"
            "DOCX (*.docx)"
        )
        filepath, _ = QFileDialog.getSaveFileName(self, "Exporter résultats", "", file_filters)
        if not filepath:
            return

        try:
            ext = os.path.splitext(filepath)[1].lower()

            if ext == '.txt':
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("RÉSULTATS D'ANALYSE STATISTIQUE\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                    for title, content in texts:
                        f.write(f"{title}\n{content}\n\n")

            elif ext == '.pdf':
                exporter = ReportExporter("Rapport StatPro", filepath)
                exporter.add_section("RÉSULTATS D'ANALYSE STATISTIQUE",
                                     f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                for title, content in texts:
                    exporter.add_section(title, content)
                exporter.export(filepath)

            elif ext == '.odt':
                from docx import Document as DocxDocument
                from io import BytesIO
                import subprocess

                doc = DocxDocument()
                doc.add_heading("RÉSULTATS D'ANALYSE STATISTIQUE", level=1)
                doc.add_paragraph(f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                for title, content in texts:
                    doc.add_heading(title, level=2)
                    doc.add_paragraph(content)
                tmp_docx = filepath.replace('.odt', '.docx')
                doc.save(tmp_docx)
                try:
                    subprocess.run(["libreoffice", "--headless", "--convert-to", "odt", tmp_docx],
                                   capture_output=True, timeout=30)
                    if os.path.exists(filepath):
                        os.remove(tmp_docx)
                    elif os.path.exists(tmp_docx):
                        import shutil
                        shutil.move(tmp_docx, filepath)
                except Exception:
                    if os.path.exists(tmp_docx):
                        os.rename(tmp_docx, filepath)

            elif ext == '.docx':
                from docx import Document
                doc = Document()
                doc.add_heading("RÉSULTATS D'ANALYSE STATISTIQUE", level=1)
                doc.add_paragraph(f"Date : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                for title, content in texts:
                    doc.add_heading(title, level=2)
                    doc.add_paragraph(content)
                doc.save(filepath)

            self.status_bar.showMessage(f"Résultats exportés : {filepath}")
        except ImportError:
            QMessageBox.warning(self, "Export", "La bibliothèque 'python-docx' n'est pas installée.\nInstallez-la avec : pip install python-docx")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'exporter :\n{e}")

    def _show_about(self):
        QMessageBox.information(
            self, "À propos",
            "StatPro - Analyse Statistique\n\n"
            "Alternative à Minitab\n\n"
            "Fonctionnalités :\n"
            "- Tableur avec Copier/Coller et menu contextuel\n"
            "- Capabilité (six-pack), Normalité, Identification distribution\n"
            "- Cartes de contrôle (X̄-R, I-MR, EWMA, CUSUM)\n"
            "- Corrélation, Régression, ANOVA, Test t\n"
            "- Boxplots, MSA/Gage R&R, Valeurs aberrantes\n"
            "- Export TXT/PDF/ODT/DOCX\n\n"
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
