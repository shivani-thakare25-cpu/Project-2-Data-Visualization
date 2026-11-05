
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_NAME = "Data Visualization Project"
OUTPUT_DIR = "data_visualization_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palettes
CUSTOM_PALETTE = ["#264653", "#2A9D8F", "#E9C46A",
                  "#F4A261", "#E76F51"]

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.0
)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def save_chart(filename):
    """Save the current chart and close its figure."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Chart saved: {filepath}")
    plt.close()

def load_dataset():
    """Load Titanic data and handle missing values."""
    print("\nLoading Titanic dataset...")

    try:
        # This dataset requires an internet connection
        data = sns.load_dataset("titanic")

        if data.empty:
            raise ValueError("The dataset is empty.")

        print("Dataset loaded successfully!")

    except Exception as error:
        print("Could not download the Titanic dataset.")
        print("Please check your internet connection.")
        print("Details:", error)
        sys.exit(1)

    data = data.drop_duplicates().copy()

    
    data["age"] = data["age"].fillna(data["age"].median())
    data["fare"] = data["fare"].fillna(data["fare"].median())

    data["embarked"] = data["embarked"].fillna("Unknown")
    data["embark_town"] = data["embark_town"].fillna("Unknown")

    print("\nDataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))

    print("\nDataset information:")
    data.info()

    print("\nMissing values after cleaning:")
    print(data.isnull().sum().to_string())

    # Save cleaned data
    data.to_csv(
        os.path.join(OUTPUT_DIR, "cleaned_titanic_data.csv"),
        index=False
    )

    return data


def create_bar_chart(data):
    """Compare passenger counts by passenger class."""

    class_order = ["First", "Second", "Third"]

    plt.figure(figsize=(9, 6))

    ax = sns.countplot(
        data=data,
        x="class",
        order=class_order,
        hue="class",
        palette=CUSTOM_PALETTE[:3],
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=10)

    plt.title("Titanic Passengers by Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.grid(axis="y", alpha=0.25)

    save_chart("01_bar_chart.png")


def create_scatter_plot(data):
    """Study the relationship between age and ticket fare."""

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=data,
        x="age",
        y="fare",
        hue="survived",
        style="sex",
        palette={0: "#E76F51", 1: "#2A9D8F"},
        alpha=0.75,
        s=65
    )

    plt.title("Passenger Age vs Ticket Fare")
    plt.xlabel("Age (years)")
    plt.ylabel("Ticket Fare")
    plt.legend(
        title="Survival / Gender",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    save_chart("02_scatter_plot.png")


def create_histogram(data):
    """Show the distribution of passenger ages."""

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=data,
        x="age",
        bins=25,
        kde=True,
        color="#2A9D8F",
        edgecolor="white",
        alpha=0.85
    )

    plt.title("Distribution of Passenger Ages")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Passengers")

    save_chart("03_histogram.png")


def create_box_plot(data):
    """Compare ticket fares across passenger classes."""

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=data,
        x="class",
        y="fare",
        order=["First", "Second", "Third"],
        hue="class",
        palette=CUSTOM_PALETTE[:3],
        legend=False,
        showfliers=True
    )

    plt.title("Ticket Fare Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Ticket Fare")

    save_chart("04_box_plot.png")


def create_correlation_matrix(data):
    """Calculate and save correlations between numeric columns."""

    numeric_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    numeric_data = data[numeric_columns]
    correlation_matrix = numeric_data.corr()

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(2).to_string())

    correlation_matrix.to_csv(
        os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    )

    return correlation_matrix


def create_heatmap(correlation_matrix):
    """Visualize correlations using a custom color map."""

    plt.figure(figsize=(11, 8))

    custom_cmap = sns.diverging_palette(
        220, 20, as_cmap=True
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.7,
        square=True,
        cbar_kws={"label": "Correlation Coefficient"}
    )

    plt.title("Titanic Dataset Correlation Heatmap")

    save_chart("05_correlation_heatmap.png")


def create_survival_chart(data):
    """Compare survival counts by gender."""

    plt.figure(figsize=(9, 6))

    ax = sns.countplot(
        data=data,
        x="sex",
        hue="survived",
        palette={0: "#E76F51", 1: "#2A9D8F"}
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Female", "Male"])

    # Use clearer legend labels
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Did not survive", "Survived"],
              title="Outcome")

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9)

    plt.title("Titanic Survival Counts by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Passengers")

    save_chart("06_survival_comparison.png")


def generate_summary(data):
    """Print useful findings from the dataset."""

    total_passengers = len(data)
    survived_count = int(data["survived"].sum())
    survival_rate = data["survived"].mean() * 100

    average_age = data["age"].mean()
    average_fare = data["fare"].mean()

    survival_by_gender = (
        data.groupby("sex", observed=True)["survived"]
        .mean() * 100
    )

    survival_by_class = (
        data.groupby("class", observed=True)["survived"]
        .mean() * 100
    )

    report = f"""
============================================================
             DATA VISUALIZATION PROJECT REPORT
============================================================

Dataset: Titanic Passenger Dataset

Total passengers analyzed : {total_passengers}
Passengers who survived   : {survived_count}
Overall survival rate     : {survival_rate:.2f}%
Average passenger age     : {average_age:.2f} years
Average ticket fare       : {average_fare:.2f}

Survival rate by gender:
{survival_by_gender.round(2).to_string()}

Survival rate by passenger class:
{survival_by_class.round(2).to_string()}

KEY OBSERVATIONS:
1. The bar chart compares the number of passengers
   across first, second and third classes.
2. The scatter plot compares passenger age and ticket fare.
3. The histogram displays the distribution of ages.
4. The box plot compares ticket fares and identifies
   possible outliers.
5. The correlation heatmap shows the strength and direction
   of linear relationships between numerical variables.
6. The survival chart compares survival counts by gender.

Note: Correlation does not prove causation.
============================================================
"""

    print(report)

    with open(
        os.path.join(OUTPUT_DIR, "project_summary.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)


def main():
    print("=" * 60)
    print(PROJECT_NAME.upper())
    print("=" * 60)

    data = load_dataset()

    create_bar_chart(data)
    create_scatter_plot(data)
    create_histogram(data)
    create_box_plot(data)

    correlation_matrix = create_correlation_matrix(data)
    create_heatmap(correlation_matrix)

    create_survival_chart(data)
    generate_summary(data)

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print("All charts and reports are in:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
