---
title: Inventory Management Policy
version: "1.3"
effective_date: 2024-01-20
department: Operations
---

# 1. Purpose
Sets reorder rules and stock-level targets for raw materials and components procured through
the standard Purchase Order process.

# 2. Reorder Point Calculation
Each SKU's reorder point is set at (average daily usage x supplier lead time) + safety stock.
Lead time is the supplier's category-level SLA target from the SLA Policy, not the supplier's
historical average -- using the SLA target keeps reorder points aligned with what Procurement is
contractually expecting, not what has merely been observed.

# 3. Safety Stock Tiers
| Supplier Risk Tier | Safety stock (days of usage) |
|---|---|
| Vendor Risk Score >= 70 | 5 |
| Vendor Risk Score 40-69 | 10 |
| Vendor Risk Score < 40 (Manual Credit Review) | 20 |

Suppliers under Manual Credit Review (Procurement Policy Section 5) carry double the standard
safety stock because their orders are more likely to be delayed by the review step itself.

# 4. Cycle Counting
Physical cycle counts are performed monthly for A-class SKUs (top 20% of usage value) and
quarterly for all others. A count variance exceeding 3% for any SKU triggers an immediate
Purchase Requisition freeze on that SKU pending investigation.

# 5. Emergency Purchase Interaction
An Emergency Purchase (Exception Handling Procedure Section 2) does not update the reorder point
calculation for that SKU -- emergency demand is treated as an anomaly, not a signal to raise the
standing reorder point, unless three or more emergency purchases occur for the same SKU within
a quarter, at which point Section 2's reorder point formula is recalculated manually.

# 6. Obsolete Stock
Stock with zero usage for 12 consecutive months is flagged obsolete and excluded from reorder
point calculations; disposal follows the Data Retention Policy's Section 4 process for physical
(non-records) asset disposal.
