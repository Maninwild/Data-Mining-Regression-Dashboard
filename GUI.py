
# COMPLETE DATA MINING DASHBOARD GUI


import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


df = pd.read_csv("defects_data.csv")
df = df.dropna()

X = df.drop("repair_cost", axis=1)
y = df["repair_cost"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Regression models: fit on RAW (unscaled) features, matching the notebook ---

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_r2 = r2_score(y_test, dt_pred)

# KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_r2 = r2_score(y_test, knn_pred)

# --- PCA / KMeans: fit on the FULL scaled dataset (all rows), matching the notebook ---
# (the notebook scales/fits PCA and KMeans on X, not just X_train)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

#GUI

root = tk.Tk()
root.title("Complete Data Mining Dashboard")
root.geometry("1200x800")
root.configure(bg="#f0f0f0")

title = tk.Label(root,
                 text="DATA MINING REGRESSION DASHBOARD",
                 font=("Arial", 20, "bold"),
                 bg="#003366",
                 fg="white")
title.pack(fill=tk.X)


notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)



# TAB 1 - MODEL PERFORMANCE


tab1 = tk.Frame(notebook, bg="white")
notebook.add(tab1, text="Model Performance")

tk.Label(tab1, text="R2 Scores",
         font=("Arial", 16, "bold"),
         bg="white").pack(pady=10)

tk.Label(tab1, text=f"Linear Regression R2: {round(lr_r2,4)}",
         bg="white").pack()

tk.Label(tab1, text=f"Decision Tree R2: {round(dt_r2,4)}",
         bg="white").pack()

tk.Label(tab1, text=f"KNN R2: {round(knn_r2,4)}",
         bg="white").pack()

fig1 = plt.Figure(figsize=(5,4))
ax1 = fig1.add_subplot(111)
models = ["LR", "DT", "KNN"]
scores = [lr_r2, dt_r2, knn_r2]
ax1.bar(models, scores)
ax1.set_title("Model Comparison (R2)")

canvas1 = FigureCanvasTkAgg(fig1, tab1)
canvas1.draw()
canvas1.get_tk_widget().pack()


tk.Label(tab1,
         text="Inference:\nHigher R2 indicates better prediction performance.\n"
              "Decision Tree captures non-linear patterns.\n"
              "Linear Regression models linear relationships.",
         bg="white").pack(pady=20)



# TAB 2 - PCA


tab2 = tk.Frame(notebook, bg="white")
notebook.add(tab2, text="PCA")

fig2 = plt.Figure(figsize=(5,4))
ax2 = fig2.add_subplot(111)
ax2.scatter(X_pca[:,0], X_pca[:,1])
ax2.set_title("PCA - 2D Projection")

canvas2 = FigureCanvasTkAgg(fig2, tab2)
canvas2.draw()
canvas2.get_tk_widget().pack()

tk.Label(tab2,
         text="Inference:\nPCA reduces dimensionality while preserving variance.\n"
              "Helps visualize high-dimensional defect data.",
         bg="white").pack(pady=20)



# TAB 3 - KNN GRAPH


tab3 = tk.Frame(notebook, bg="white")
notebook.add(tab3, text="KNN Analysis")

k_vals = range(1, 15)
r2_vals = []

for k in k_vals:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2_vals.append(r2_score(y_test, pred))

fig3 = plt.Figure(figsize=(5,4))
ax3 = fig3.add_subplot(111)
ax3.plot(k_vals, r2_vals)
ax3.set_title("KNN - K vs R2")
ax3.set_xlabel("K")
ax3.set_ylabel("R2")

canvas3 = FigureCanvasTkAgg(fig3, tab3)
canvas3.draw()
canvas3.get_tk_widget().pack()

tk.Label(tab3,
         text="Inference:\nOptimal K is where R2 is highest.\n"
              "Too small K → overfitting.\n"
              "Too large K → underfitting.",
         bg="white").pack(pady=20)



# TAB 4 - KMEANS


tab4 = tk.Frame(notebook, bg="white")
notebook.add(tab4, text="KMeans Clustering")

fig4 = plt.Figure(figsize=(5,4))
ax4 = fig4.add_subplot(111)
ax4.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
ax4.set_title("KMeans Clusters (PCA Projection)")

canvas4 = FigureCanvasTkAgg(fig4, tab4)
canvas4.draw()
canvas4.get_tk_widget().pack()

tk.Label(tab4,
         text="Inference:\nKMeans groups defects into clusters\n"
              "based on similarity in feature space.",
         bg="white").pack(pady=20)



# TAB 5 - OLAP


tab5 = tk.Frame(notebook, bg="white")
notebook.add(tab5, text="OLAP")

olap_data = df.groupby("severity")["repair_cost"].mean()

fig5 = plt.Figure(figsize=(5,4))
ax5 = fig5.add_subplot(111)
ax5.bar(olap_data.index, olap_data.values)
ax5.set_title("Average Repair Cost by Severity")

canvas5 = FigureCanvasTkAgg(fig5, tab5)
canvas5.draw()
canvas5.get_tk_widget().pack()

tk.Label(tab5,
         text="Inference:\nOLAP aggregates data for business insights.\n"
              "Helps analyze cost patterns across severity levels.",
         bg="white").pack(pady=20)


root.mainloop()