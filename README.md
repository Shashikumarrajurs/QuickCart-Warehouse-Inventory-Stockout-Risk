# QuickCart-Warehouse-Inventory-Stockout-Risk
# 🛒 QuickCart Warehouse Inventory Stockout Risk

### Supervised Machine Learning Classification Project

> Predict stockout risk for every SKU × Store × Day as `Safe`, `At-Risk`, or `Imminent`.

---

## 📌 Project Overview

Quick-commerce warehouses need to know which products are likely to run out before the next supplier delivery arrives.

This project builds a **3-class supervised machine learning classification system** to identify daily inventory risk:

| Class | Meaning |
|---|---|
| 🟢 **Safe** | Stock cover exceeds replenishment wait by 3+ days |
| 🟡 **At-Risk** | Stock cover is within 3 days of replenishment wait |
| 🔴 **Imminent** | Stock may run out before the next delivery |

The project combines:

- Inventory levels
- Sales velocity
- Supplier reliability
- Product attributes
- Store characteristics
- Festival demand
- Promotional events

The project specification defines the target as **Safe / At-Risk / Imminent** for every product, store and day. 

---

## 🎯 Business Objective

The main objective is to help an inventory team:

- 🚨 Identify products with **Imminent** stockout risk
- ⚠️ Identify **At-Risk** products early
- 📦 Support replenishment planning
- 🎉 Account for festival-driven demand spikes
- 🚚 Consider supplier reliability and lead times
- 💰 Reduce lost sales caused by stockouts
- 🏪 Improve inventory availability across stores

---

# 🗂️ Dataset

The project contains five relational datasets.

| Dataset | Rows | Purpose |
|---|---:|---|
| `dim_stores.csv` | 12 | Store dimension |
| `dim_skus.csv` | 60 | Product/SKU dimension |
| `dim_suppliers.csv` | 15 | Supplier dimension |
| `dim_events.csv` | 30 | Festival/promotion calendar |
| `fact_inventory_daily.csv` | 21,600 | Main modeling table |

### Target Distribution

| Risk Class | Rows | Percentage |
|---|---:|---:|
| Safe | 14,131 | 65.42% |
| At-Risk | 5,186 | 24.01% |
| Imminent | 2,283 | 10.57% |

The dataset is therefore naturally imbalanced.

Because the **Imminent** class represents the most operationally costly failure, the project gives special attention to its recall.

---

# 🔗 Data Model

```text
                  ┌─────────────────┐
                  │  dim_stores.csv │
                  └────────┬────────┘
                           │
                           │
┌────────────────┐         │        ┌──────────────────┐
│ dim_skus.csv   │─────────┼────────│ dim_suppliers.csv│
└────────────────┘         │        └──────────────────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │ fact_inventory_daily.csv│
                │       21,600 rows       │
                └────────────┬────────────┘
                             │
                             │
                    ┌────────▼────────┐
                    │ dim_events.csv  │
                    └────────┬────────┘
                             │
                             ▼
                    Feature Engineering
                             │
                             ▼
                       ML Classification
                             │
                             ▼
                Safe / At-Risk / Imminent
