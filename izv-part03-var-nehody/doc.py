"""
@file doc.py
@brief: Documentation utilities for collision data analysis
@author: Martin Valapka - xvalapm00
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

df_accidents = pd.read_pickle("accidents.pkl.gz")
df_consequences = pd.read_pickle("consequences.pkl.gz")

p2_datetime = pd.to_datetime(
    df_accidents["p2a"] + ' ' + df_accidents["p2b"].astype(str).str.zfill(4),
    format="%d.%m.%Y %H%M",
    errors="coerce"
)

df_accidents["month"] = p2_datetime.dt.month
df_accidents["year"] = p2_datetime.dt.year
df_accidents["hour"] = p2_datetime.dt.hour
df_accidents["weekday"] = p2_datetime.dt.dayofweek
df_accidents["day_name"] = p2_datetime.dt.day_name()

# Filter data for years 2023 and 2024
df_accidents = df_accidents[df_accidents["year"].isin([2023, 2024])].reset_index(drop=True)

merged = df_accidents.merge(df_consequences[["p1", "p59g"]], on="p1", how="left")

severity_map = {1: "Fatal", 2: "Serious", 3: "Minor", 4: "No Injury"}
merged["severity_label"] = merged["p59g"].map(severity_map).fillna("Unknown")
merged["is_severe"] = merged["p59g"].isin([1, 2])
merged["is_fatal"] = merged["p59g"] == 1

merged["has_consequences"] = merged["p59g"].isin([1, 2, 3])

def hour_to_period(h):
    """
    @brief Convert hour to period of day
    @param h: Hour of the day (0-23)
    @return: Period of day as string
    """
    h = int(h)
    if 0 <= h < 6:
        return "Night"
    elif 6 <= h < 12:
        return "Morning"
    elif 12 <= h < 18:
        return "Afternoon"
    elif 18 <= h < 24:
        return "Evening"
    return None

merged["period"] = merged["hour"].apply(hour_to_period)

period_fatal = merged[merged["period"].notna() & merged["is_fatal"]].groupby("period").size()
period_total = merged[merged["period"].notna()].groupby("period").size()
period_fatal_percent = (period_fatal / period_total * 100).fillna(0)

by_hour_severity = merged[merged["hour"].notna()].groupby(["hour", "severity_label"]).size().unstack(fill_value=0)
by_hour_severity = by_hour_severity.reindex(["Fatal", "Serious", "Minor", "No Injury", "Unknown"], axis=1, fill_value=0)
by_hour_all = merged[merged["hour"].notna()].groupby("hour").size()
by_hour_all = by_hour_all.reindex(range(24), fill_value=0)

by_hour_fatal = merged[merged["hour"].notna() & merged["is_fatal"]].groupby("hour").size().reindex(range(24), fill_value=0)
by_hour_fatal_percent = (by_hour_fatal / by_hour_all * 100).fillna(0)

# Contingency table for period-based chi-square test
period_contingency_table = pd.DataFrame({
    "Fatal": period_fatal,
    "Not Fatal": (period_total - period_fatal)
}).fillna(0)
chi2, p_value, dof, expected = chi2_contingency(period_contingency_table)
period_contingency_table.to_csv("contingency_table_periods.csv", index_label="period")

def plot_severity_distribution(save_fig=False):
    """
    @brief Plot the distribution of accident severity by hour
    @param save_fig: Boolean flag to save the figure as a PNG file
    """

    fig, ax1 = plt.subplots(figsize=(14, 6))

    by_hour_severity.plot(kind="bar", stacked=True, ax=ax1, colormap="viridis", width=0.8)
    ax1.set_title("Accident Severity Distribution by Hour", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Hour of the Day")
    ax1.set_ylabel("Number of Accidents")
    ax1.set_xticks(np.arange(0, 24))
    ax1.set_xticklabels([f"{int(i)}" for i in range(24)], rotation=0)
    ax1.grid(alpha=0.3, axis='y')
    
    ax1.legend(title="Severity", loc="upper right", fontsize=10, title_fontsize=11)

    plt.tight_layout()

    if save_fig:
        fig.savefig("severity_by_hour.png", dpi=300, bbox_inches="tight")
        fig.savefig("severity_by_hour.pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plot_serious_percentage(save_fig=False):
    """
    @brief Plot the percentage of fatal accidents by period of day
    @param save_fig: Boolean flag to save the figure as a PNG file
    """

    fig, ax2 = plt.subplots(figsize=(10, 6))

    periods = ["Night", "Morning", "Afternoon", "Evening"]
    period_pcts = [period_fatal_percent.get(p, 0) for p in periods]
    mean_pct = period_fatal_percent.mean()
    colors = ['#d62728' if pct > mean_pct else '#1f77b4' for pct in period_pcts]
    
    bars = ax2.bar(periods, period_pcts, color=colors, edgecolor="black", linewidth=1, width=0.6)
    
    for i, (period, pct) in enumerate(zip(periods, period_pcts)):
        ax2.annotate(f"{pct:.2f}%", xy=(i, pct), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)
    
    ax2.text(0.98, 0.97, f"Average: {mean_pct:.2f}%", transform=ax2.transAxes,
             fontsize=11, verticalalignment="top", horizontalalignment="right",
             bbox=dict(facecolor="white", alpha=0.5))
    ax2.set_title("Percentage of Fatal Accidents by Period of Day", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Period of Day")
    ax2.set_ylabel("Percentage of Fatal Accidents (%)")
    ax2.set_ylim(0, max(period_pcts) * 1.2 if period_pcts else 1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_fig:
        fig.savefig("fatal_percentage_by_period.png", dpi=300, bbox_inches="tight")
        fig.savefig("fatal_percentage_by_period.pdf", bbox_inches="tight")
        fig.savefig("fig.pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)

def print_chi2_results():
    """
    @brief Print the results of the chi-squared test for fatal independence from period of day
    """

    print("\n" + "="*70)
    print("CHI-SQUARED TEST: Fatal Accident Independence from Period of Day")
    print("="*70)
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Degrees of Freedom: {dof}")
    print(f"\nInterpretation:")
    if p_value < 0.05:
        print(f"SIGNIFICANT (p < 0.05): Fatal accident rate IS dependent on period of day")
    else:
        print(f"NOT SIGNIFICANT (p >= 0.05): No significant relationship")
    print("="*70 + "\n")

def print_severity_table():
    """
    @brief Print a formatted table of fatal accidents by hour
    """

    h_width = 4
    v_width = 8
    t_width = 8

    header = (
        f"{'Hour':<{h_width}} | "
        f"{'Fatal':>{v_width}} | "
        f"{'Total':>{t_width}} | "
        f"{'% Fatal':>8}"
    )

    print("\n" + "=" * len(header))
    print("TABLE: Fatal Accidents by Hour of Day")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for hour in range(24):
        fatal_count = by_hour_fatal.get(hour, 0)
        total_count = by_hour_all.get(hour, 0)
        fatal_pct = (fatal_count / total_count * 100) if total_count > 0 else 0
        
        print(
            f"{int(hour):02d}   | "
            f"{int(fatal_count):>{v_width}} | "
            f"{int(total_count):>{t_width}} | "
            f"{fatal_pct:>7.2f}%"
        )
    
    print("=" * len(header) + "\n")

def print_period_table():
    """
    @brief Print a formatted table of fatal accidents by period of day
    """

    p_width = 12
    f_width = 8
    t_width = 8

    header = (
        f"{'Period':<{p_width}} | "
        f"{'Fatal':>{f_width}} | "
        f"{'Total':>{t_width}} | "
        f"{'% Fatal':>8}"
    )

    print("\n" + "=" * len(header))
    print("TABLE: Fatal Accidents by Period of Day")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for period in ["Night", "Morning", "Afternoon", "Evening"]:
        fatal_count = period_fatal.get(period, 0)
        total_count = period_total.get(period, 0)
        fatal_pct = (fatal_count / total_count * 100) if total_count > 0 else 0
        
        print(
            f"{period:<{p_width}} | "
            f"{int(fatal_count):>{f_width}} | "
            f"{int(total_count):>{t_width}} | "
            f"{fatal_pct:>7.2f}%"
        )
    
    print("=" * len(header) + "\n")

def print_metrics():
    """
    @brief Print key calculated metrics for the report
    """

    print("\n" + "="*70)
    print("CALCULATED METRICS FOR REPORT")
    print("="*70)
    
    # Overall fatal accident rate
    overall_fatal_rate = (merged["is_fatal"].sum() / len(merged) * 100)
    print(f"Overall fatal accident rate: {overall_fatal_rate:.2f}%")
    
    # Peak period fatality
    peak_period = period_fatal_percent.idxmax()
    peak_period_fatal = period_fatal_percent.max()
    print(f"Peak period for fatal accidents: {peak_period} with {peak_period_fatal:.2f}% fatal")
    
    # Lowest period fatality
    lowest_period = period_fatal_percent.idxmin()
    lowest_period_fatal = period_fatal_percent.min()
    print(f"Safest period: {lowest_period} with {lowest_period_fatal:.2f}% fatal")
    
    # Ratio between peak and lowest
    fatal_ratio = peak_period_fatal / lowest_period_fatal if lowest_period_fatal > 0 else 0
    print(f"Peak/Lowest fatal ratio: {fatal_ratio:.2f}x")
    
    # Total accidents analyzed
    total_accidents = len(merged)
    print(f"Total accidents analyzed (2023-2024): {total_accidents:,}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    plot_severity_distribution(save_fig=True)
    plot_serious_percentage(save_fig=True)
    print_severity_table()
    print_period_table()
    print_metrics()
    print_chi2_results()
