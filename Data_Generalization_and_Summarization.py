"""
Data Mining Lab - Experiment 6: Data Generalization and Summarization
Concept hierarchy (City -> District -> State), aggregation and summaries.
"""

import pandas as pd
import matplotlib.pyplot as plt

data = [
    {"City": "Pune", "District": "Pune", "State": "Maharashtra", "Department": "IT", "Sales": 12000},
    {"City": "Pune", "District": "Pune", "State": "Maharashtra", "Department": "CS", "Sales": 15000},
    {"City": "Mumbai", "District": "Mumbai Suburban", "State": "Maharashtra", "Department": "IT", "Sales": 18000},
    {"City": "Mumbai", "District": "Mumbai Suburban", "State": "Maharashtra", "Department": "CS", "Sales": 14000},
    {"City": "Nagpur", "District": "Nagpur", "State": "Maharashtra", "Department": "IT", "Sales": 9000},
    {"City": "Nagpur", "District": "Nagpur", "State": "Maharashtra", "Department": "CS", "Sales": 11000}
]
df = pd.DataFrame(data)
print("DETAILED DATA")
print(df.to_string(index=False))
print("\nGENERALIZED STATE-LEVEL SUMMARY")
print(df.groupby("State", as_index=False)["Sales"].sum().to_string(index=False))
print("\nCITY-LEVEL AGGREGATION")
city_summary = df.groupby("City", as_index=False)["Sales"].sum()
print(city_summary.to_string(index=False))
print("\nDEPARTMENT SUMMARY")
print(df.groupby("Department", as_index=False)["Sales"].sum().to_string(index=False))
print("\nCOMPACT SUMMARY (City, total sales, record count)")
print(df.groupby("City").agg(Total_Sales=("Sales", "sum"),
      Record_Count=("Sales", "count")).reset_index().to_string(index=False))

city_summary.plot(x="City", y="Sales", kind="bar", legend=False)
plt.title("Aggregated Sales by City")
plt.xlabel("City")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("23610070_A6_city_summary.png", dpi=150)
plt.show()
