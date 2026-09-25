import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# =========================================================
# DATASET
# =========================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target_names[iris.target]


# =========================================================
# COLOURS - DIFFERENT TEMPLATE
# =========================================================

BG = "#f3efe7"          # Cream background
DARK = "#263238"        # Charcoal
TEAL = "#00695c"        # Teal
TEAL_LIGHT = "#80cbc4"
CREAM = "#fffdf8"
GOLD = "#c28b2c"
MUTED = "#6b7280"
WHITE = "#ffffff"
BORDER = "#d6d0c4"


# =========================================================
# PLOT FUNCTIONS
# =========================================================

def show_bar():
    plt.figure(figsize=(6, 4))
    df["species"].value_counts().plot(kind="bar")
    plt.title("Species Count")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def show_histogram():
    plt.figure(figsize=(6, 4))
    plt.hist(df["sepal length (cm)"], bins=10)
    plt.title("Distribution of Sepal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def show_scatter():
    plt.figure(figsize=(6, 4))

    sns.scatterplot(
        data=df,
        x="sepal length (cm)",
        y="petal length (cm)",
        hue="species"
    )

    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.tight_layout()
    plt.show()


def show_box():
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df,
        x="species",
        y="petal length (cm)"
    )

    plt.title("Petal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.tight_layout()
    plt.show()


def show_pie():
    plt.figure(figsize=(6, 4))

    df["species"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Species Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def show_heatmap():
    plt.figure(figsize=(6, 4))

    numeric_df = df.drop(columns=["species"])

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="YlGnBu",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# =========================================================
# DATASET PREVIEW
# =========================================================

def show_preview():
    output.delete("1.0", tk.END)

    output.insert(
        tk.END,
        "IRIS DATASET — FIRST 10 RECORDS\n"
        + "-" * 85
        + "\n\n"
    )

    output.insert(
        tk.END,
        df.head(10).to_string(index=True)
    )

    output.insert(
        tk.END,
        "\n\n\nDataset Shape : "
        + str(df.shape)
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

def show_information():
    output.delete("1.0", tk.END)

    output.insert(
        tk.END,
        "DATASET INFORMATION\n"
        + "-" * 85
        + "\n\n"
    )

    output.insert(
        tk.END,
        "Dataset Name : Iris Dataset\n"
        "Records      : 150\n"
        "Features     : 4\n"
        "Target       : Species\n\n"
        "FEATURES\n"
        "--------\n"
        "1. Sepal Length\n"
        "2. Sepal Width\n"
        "3. Petal Length\n"
        "4. Petal Width\n\n"
        "SPECIES\n"
        "-------\n"
        "Setosa\n"
        "Versicolor\n"
        "Virginica\n"
    )


def clear_output():
    output.delete("1.0", tk.END)


def exit_application():
    root.destroy()


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title("Data Studio | Iris Dataset")
root.geometry("800x560")
root.resizable(False, False)
root.configure(bg=BG)


# =========================================================
# TOP HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=DARK,
    height=82
)

header.pack(
    fill="x"
)

header.pack_propagate(False)


tk.Label(
    header,
    text="DATA STUDIO",
    bg=DARK,
    fg=TEAL_LIGHT,
    font=("Segoe UI", 10, "bold")
).place(
    x=25,
    y=13
)


tk.Label(
    header,
    text="Iris Dataset Explorer",
    bg=DARK,
    fg=WHITE,
    font=("Segoe UI", 21, "bold")
).place(
    x=23,
    y=32
)


tk.Label(
    header,
    text="PLOTTING & VISUAL ANALYSIS",
    bg=DARK,
    fg="#cfd8dc",
    font=("Segoe UI", 8, "bold")
).place(
    x=590,
    y=36
)


# =========================================================
# MAIN CONTENT
# =========================================================

content = tk.Frame(
    root,
    bg=BG
)

content.pack(
    fill="both",
    expand=True,
    padx=18,
    pady=14
)


# =========================================================
# LEFT NAVIGATION
# =========================================================

nav = tk.Frame(
    content,
    bg=DARK,
    width=170
)

nav.pack(
    side="left",
    fill="y",
    padx=(0, 12)
)

nav.pack_propagate(False)


tk.Label(
    nav,
    text="MENU",
    bg=DARK,
    fg=TEAL_LIGHT,
    font=("Segoe UI", 9, "bold")
).pack(
    anchor="w",
    padx=18,
    pady=(20, 12)
)


def nav_button(text, command, active=False):

    color = TEAL if active else DARK

    button = tk.Button(
        nav,
        text=text,
        command=command,
        bg=color,
        fg=WHITE,
        activebackground=TEAL,
        activeforeground=WHITE,
        relief="flat",
        bd=0,
        anchor="w",
        padx=15,
        font=("Segoe UI", 9, "bold"),
        cursor="hand2"
    )

    button.pack(
        fill="x",
        padx=10,
        pady=3,
        ipady=7
    )


nav_button("Overview", show_information, True)
nav_button("Dataset Preview", show_preview)


tk.Label(
    nav,
    text="PLOTS",
    bg=DARK,
    fg=TEAL_LIGHT,
    font=("Segoe UI", 9, "bold")
).pack(
    anchor="w",
    padx=18,
    pady=(18, 8)
)


nav_button("Bar Chart", show_bar)
nav_button("Histogram", show_histogram)
nav_button("Scatter Plot", show_scatter)
nav_button("Box Plot", show_box)
nav_button("Pie Chart", show_pie)
nav_button("Heatmap", show_heatmap)


# =========================================================
# RIGHT SIDE
# =========================================================

right = tk.Frame(
    content,
    bg=BG
)

right.pack(
    side="right",
    fill="both",
    expand=True
)


# =========================================================
# OVERVIEW TITLE
# =========================================================

tk.Label(
    right,
    text="Dataset Overview",
    bg=BG,
    fg=DARK,
    font=("Segoe UI", 16, "bold")
).pack(
    anchor="w"
)


tk.Label(
    right,
    text="A compact view of the dataset used for visualization.",
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(
    anchor="w",
    pady=(2, 10)
)


# =========================================================
# STATISTIC CARDS
# =========================================================

stats = tk.Frame(
    right,
    bg=BG
)

stats.pack(
    fill="x"
)


def stat_box(value, title, accent):

    box = tk.Frame(
        stats,
        bg=CREAM,
        width=125,
        height=65,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    box.pack(
        side="left",
        padx=(0, 7)
    )

    box.pack_propagate(False)

    tk.Frame(
        box,
        bg=accent,
        width=5
    ).pack(
        side="left",
        fill="y"
    )

    tk.Label(
        box,
        text=value,
        bg=CREAM,
        fg=DARK,
        font=("Segoe UI", 16, "bold")
    ).pack(
        pady=(8, 0)
    )

    tk.Label(
        box,
        text=title,
        bg=CREAM,
        fg=MUTED,
        font=("Segoe UI", 8)
    ).pack()


stat_box("150", "Records", TEAL)
stat_box("4", "Features", GOLD)
stat_box("3", "Species", "#8e44ad")
stat_box("Iris", "Dataset", "#d35400")


# =========================================================
# DATA TABLE AREA
# =========================================================

table_card = tk.Frame(
    right,
    bg=CREAM,
    highlightbackground=BORDER,
    highlightthickness=1
)

table_card.pack(
    fill="both",
    expand=True,
    pady=(12, 10)
)


tk.Label(
    table_card,
    text="Data Preview",
    bg=CREAM,
    fg=DARK,
    font=("Segoe UI", 11, "bold")
).pack(
    anchor="w",
    padx=12,
    pady=(10, 2)
)


tk.Label(
    table_card,
    text="Click Dataset Preview from the menu to load records.",
    bg=CREAM,
    fg=MUTED,
    font=("Segoe UI", 8)
).pack(
    anchor="w",
    padx=12
)


output = ScrolledText(
    table_card,
    wrap=tk.NONE,
    bg="#faf9f5",
    fg=DARK,
    font=("Consolas", 8),
    relief="flat",
    borderwidth=0
)

output.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=8
)


# =========================================================
# DEFAULT CONTENT
# =========================================================

output.insert(
    tk.END,
    "IRIS DATASET\n"
    + "-" * 85
    + "\n\n"
    "Use the left menu to explore the dataset.\n\n"
    "Available visualizations:\n\n"
    "• Bar Chart\n"
    "• Histogram\n"
    "• Scatter Plot\n"
    "• Box Plot\n"
    "• Pie Chart\n"
    "• Correlation Heatmap"
)


# =========================================================
# BOTTOM ACTION BAR
# =========================================================

bottom = tk.Frame(
    root,
    bg=BG
)

bottom.pack(
    fill="x",
    padx=18,
    pady=(0, 10)
)


tk.Label(
    bottom,
    text="Iris Dataset  |  Data Visualization Project",
    bg=BG,
    fg=MUTED,
    font=("Segoe UI", 8)
).pack(
    side="left"
)


tk.Button(
    bottom,
    text="Clear",
    command=clear_output,
    bg="#d6d0c4",
    fg=DARK,
    activebackground="#c8c0b2",
    relief="flat",
    font=("Segoe UI", 8, "bold"),
    width=8,
    cursor="hand2"
).pack(
    side="right",
    padx=3
)


tk.Button(
    bottom,
    text="Exit",
    command=exit_application,
    bg=DARK,
    fg=WHITE,
    activebackground=TEAL,
    relief="flat",
    font=("Segoe UI", 8, "bold"),
    width=8,
    cursor="hand2"
).pack(
    side="right",
    padx=3
)


# =========================================================
# START
# =========================================================

root.mainloop()