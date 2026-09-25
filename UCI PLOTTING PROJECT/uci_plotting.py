import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris


# -----------------------------------
# 1. Load UCI-style Iris Dataset
# -----------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target_names[iris.target]


# -----------------------------------
# 2. Display Dataset
# -----------------------------------

print("\n========== IRIS DATASET ==========\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSpecies Count:")
print(df["species"].value_counts())


# -----------------------------------
# 3. Bar Chart
# -----------------------------------

plt.figure(figsize=(7, 5))

species_count = df["species"].value_counts()

species_count.plot(kind="bar")

plt.title("Species Count")
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# -----------------------------------
# 4. Histogram
# -----------------------------------

plt.figure(figsize=(7, 5))

plt.hist(
    df["sepal length (cm)"],
    bins=10
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# -----------------------------------
# 5. Scatter Plot
# -----------------------------------

plt.figure(figsize=(7, 5))

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


# -----------------------------------
# 6. Box Plot
# -----------------------------------

plt.figure(figsize=(7, 5))

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


# -----------------------------------
# 7. Pie Chart
# -----------------------------------

plt.figure(figsize=(7, 5))

species_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Species Distribution")
plt.ylabel("")

plt.tight_layout()
plt.show()


# -----------------------------------
# 8. Correlation Heatmap
# -----------------------------------

plt.figure(figsize=(7, 5))

numeric_df = df.drop(columns=["species"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


# -----------------------------------
# 9. End Message
# -----------------------------------

print("\n===================================")
print("All plots generated successfully!")
print("===================================")