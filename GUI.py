"""Tkinter interface for the reusable data-mining analysis pipeline."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analysis import run_analysis


def add_figure(parent: tk.Widget, figure: Figure) -> None:
    canvas = FigureCanvasTkAgg(figure, parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=16, pady=12)


def add_note(parent: tk.Widget, text: str) -> None:
    tk.Label(
        parent,
        text=text,
        bg="white",
        fg="#334155",
        justify=tk.LEFT,
        wraplength=960,
        font=("Arial", 10),
    ).pack(fill=tk.X, padx=20, pady=(0, 16))


class Dashboard:
    def __init__(self, root: tk.Tk, results: dict[str, object]) -> None:
        self.root = root
        self.results = results
        self.root.title("Data Mining and Regression Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)
        self.root.configure(bg="#f1f5f9")

        tk.Label(
            root,
            text="DATA MINING & REGRESSION DASHBOARD",
            font=("Arial", 20, "bold"),
            bg="#0f3d63",
            fg="white",
            pady=16,
        ).pack(fill=tk.X)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self._model_tab(notebook)
        self._pca_tab(notebook)
        self._knn_tab(notebook)
        self._cluster_tab(notebook)
        self._olap_tab(notebook)

    @staticmethod
    def _tab(notebook: ttk.Notebook, title: str) -> tk.Frame:
        frame = tk.Frame(notebook, bg="white")
        notebook.add(frame, text=title)
        return frame

    def _model_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Model Performance")
        metrics = self.results["metrics"]
        figure = Figure(figsize=(8, 4.5), tight_layout=True)
        axis = figure.add_subplot(111)
        colors = ["#94a3b8", "#2563eb", "#0f766e", "#7c3aed"]
        axis.bar(metrics["model"], metrics["r2"], color=colors)
        axis.axhline(0, color="#0f172a", linewidth=0.8)
        axis.set_ylabel("Held-out R²")
        axis.set_title("Regression models compared with a mean baseline")
        axis.tick_params(axis="x", rotation=12)
        add_figure(tab, figure)

        rows = [
            f"{row.model}: R² {row.r2:.3f} · MAE ₹{row.mae:.2f} · RMSE ₹{row.rmse:.2f}"
            for row in metrics.itertuples()
        ]
        best = metrics.loc[metrics["r2"].idxmax()]
        add_note(
            tab,
            "\n".join(rows)
            + f"\n\nBest held-out R²: {best['model']} ({best['r2']:.3f}). "
            "A value near or below zero means the available features do not predict repair cost better than the mean baseline.",
        )

    def _pca_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "PCA")
        projection = self.results["projection"]
        explained = self.results["explained_variance"]
        figure = Figure(figsize=(7, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        axis.scatter(projection[:, 0], projection[:, 1], s=18, alpha=0.65, color="#2563eb")
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
        axis.set_title("Two-dimensional PCA projection")
        add_figure(tab, figure)
        add_note(
            tab,
            f"The first two components explain {(sum(explained) * 100):.1f}% of encoded-feature variance. "
            "This plot is exploratory and is not evidence of predictive accuracy.",
        )

    def _knn_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "KNN Selection")
        curve = self.results["knn_curve"]
        best_k = self.results["best_k"]
        figure = Figure(figsize=(7, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        axis.plot(curve["k"], curve["mean_cv_r2"], marker="o", color="#7c3aed")
        axis.axvline(best_k, color="#ef4444", linestyle="--", label=f"Selected k={best_k}")
        axis.set_xlabel("Neighbours (k)")
        axis.set_ylabel("Mean 5-fold CV R²")
        axis.set_title("KNN hyperparameter selection on training data")
        axis.legend()
        add_figure(tab, figure)
        add_note(tab, "Scaling and cross-validation are applied before KNN is evaluated on the held-out test set.")

    def _cluster_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "KMeans Clustering")
        projection = self.results["projection"]
        clusters = self.results["clusters"]
        figure = Figure(figsize=(7, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        axis.scatter(projection[:, 0], projection[:, 1], c=clusters, s=18, alpha=0.7, cmap="viridis")
        axis.set_xlabel("Principal component 1")
        axis.set_ylabel("Principal component 2")
        axis.set_title("Three KMeans clusters shown in PCA space")
        add_figure(tab, figure)
        add_note(tab, "KMeans operates on the full standardized feature matrix; PCA is used only to display the clusters.")

    def _olap_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "OLAP Summary")
        summary = self.results["olap"]
        figure = Figure(figsize=(7, 5), tight_layout=True)
        axis = figure.add_subplot(111)
        axis.bar(summary.index, summary["average_cost"], color="#0f766e")
        axis.set_ylabel("Average repair cost (₹)")
        axis.set_title("Average repair cost by severity")
        add_figure(tab, figure)
        lines = [
            f"{severity}: {int(row.defects)} defects · average ₹{row.average_cost:.2f} · total ₹{row.total_cost:.2f}"
            for severity, row in summary.iterrows()
        ]
        add_note(tab, "\n".join(lines))


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        results = run_analysis()
    except Exception as error:  # GUI boundary: show a readable startup failure.
        messagebox.showerror("Dashboard startup failed", str(error), parent=root)
        root.destroy()
        return
    root.deiconify()
    Dashboard(root, results)
    root.mainloop()


if __name__ == "__main__":
    main()
