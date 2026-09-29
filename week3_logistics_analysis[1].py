"""
Week 3 - Advanced Data Analysis and Visualization in Logistics

The script loads the hypothetical logistics dataset, calculates descriptive
statistics and correlations, and creates six logistics visualizations.

Run:
    pip install -r requirements.txt
    python week3_logistics_analysis.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/hypothetical_logistics_dataset.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["shipment_date"])

# Basic descriptive statistics
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df[[
    "shipment_volume", "distance_km", "delivery_days",
    "transport_cost", "handling_cost", "total_cost"
]].describe().round(2))

# Central tendency
print("\nMean delivery days:", round(df["delivery_days"].mean(), 2))
print("Median delivery days:", round(df["delivery_days"].median(), 2))
print("Mean total cost:", round(df["total_cost"].mean(), 2))
print("Median total cost:", round(df["total_cost"].median(), 2))

# Logistics KPI summary
print("\n=== KPI SUMMARY ===")
print("Late delivery rate (%):",
      round(df["late_delivery"].mean() * 100, 2))
print("Average shipment volume:",
      round(df["shipment_volume"].mean(), 2))

# Group analysis
mode_summary = df.groupby("transport_mode").agg(
    shipments=("shipment_date", "size"),
    avg_delivery_days=("delivery_days", "mean"),
    avg_cost=("total_cost", "mean"),
    late_rate=("late_delivery", "mean")
).reset_index()
mode_summary["late_rate"] *= 100
print("\n=== TRANSPORT MODE SUMMARY ===")
print(mode_summary.round(2))

# Correlations
corr = df[[
    "shipment_volume", "distance_km", "delivery_days",
    "transport_cost", "handling_cost", "total_cost", "late_delivery"
]].corr()
print("\n=== CORRELATION MATRIX ===")
print(corr.round(2))

# Visualization 1: delivery distribution
plt.figure(figsize=(8.5, 5))
plt.hist(df["delivery_days"], bins=24, edgecolor="black")
plt.axvline(df["delivery_days"].mean(), linestyle="--",
            label="Mean")
plt.axvline(df["delivery_days"].median(), linestyle=":",
            label="Median")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of shipments")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "01_delivery_distribution.png", dpi=180)
plt.close()

# Visualization 2: transport comparison
summary = mode_summary.sort_values("avg_delivery_days")
plt.figure(figsize=(8.5, 5))
plt.bar(summary["transport_mode"], summary["avg_delivery_days"])
plt.title("Average Delivery Time by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("Average delivery time (days)")
plt.tight_layout()
plt.savefig(OUT / "02_delivery_by_transport.png", dpi=180)
plt.close()

# Visualization 3: regional cost
region = df.groupby("region")["total_cost"].mean().sort_values(ascending=False)
plt.figure(figsize=(8.5, 5))
plt.bar(region.index, region.values)
plt.title("Average Logistics Cost by Region")
plt.xlabel("Region")
plt.ylabel("Average total cost")
plt.tight_layout()
plt.savefig(OUT / "03_cost_by_region.png", dpi=180)
plt.close()

# Visualization 4: distance vs delivery
plt.figure(figsize=(8.5, 5))
plt.scatter(df["distance_km"], df["delivery_days"], alpha=0.55)
plt.title("Relationship Between Distance and Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery time (days)")
plt.tight_layout()
plt.savefig(OUT / "04_distance_vs_delivery.png", dpi=180)
plt.close()

# Visualization 5: monthly volume
df["month"] = df["shipment_date"].dt.to_period("M").astype(str)
monthly = df.groupby("month")["shipment_volume"].sum()
plt.figure(figsize=(9, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Shipment Volume")
plt.xlabel("Month")
plt.ylabel("Shipment volume (units)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT / "05_monthly_volume.png", dpi=180)
plt.close()

# Visualization 6: correlation matrix
corr_cols = [
    "shipment_volume", "distance_km", "delivery_days",
    "transport_cost", "handling_cost", "total_cost", "late_delivery"
]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr.values, aspect="auto")
ax.set_xticks(range(len(corr_cols)), labels=corr_cols,
              rotation=45, ha="right")
ax.set_yticks(range(len(corr_cols)), labels=corr_cols)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)

ax.set_title("Correlation Matrix of Key Logistics Variables")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "06_correlation_matrix.png", dpi=180)
plt.close()

print("\nVisualizations saved in the outputs folder.")
