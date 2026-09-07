---
title: Procurement Policy
version: "2.3"
effective_date: 2024-01-15
department: Procurement
---

# 1. Purpose
This policy governs how Northstar Manufacturing (fictional company -- see the parent project's
docs/data-contract.md) creates, approves, and processes purchase orders.

# 2. Scope
Applies to all purchase requisitions across all business units and spend categories.

# 3. Purchase Requisition
All purchases must originate from an approved Purchase Requisition. Requisitions below $500 may
be self-approved by the requesting department manager. Requisitions above $500 require Procurement
review before a Purchase Order is issued.

# 4. Approval Requirements
## 4.1 Standard Approval
Purchase Orders up to $10,000 require single-level approval from the requesting department's
budget owner.

## 4.2 Secondary Approval
Purchase Orders above $10,000, or any order flagged as high-risk by the credit/vendor risk
screening, require a second approval from Procurement leadership before the order is released to
the supplier. This is the primary source of processing delay for high-value orders -- see the
companion operations-performance project's SLA-breach driver analysis.

# 5. Manual Credit Review
Orders from a new supplier (fewer than 3 prior completed orders) or from a supplier with a
below-threshold vendor risk score automatically route to Manual Credit Review regardless of order
value. Manual Credit Review is performed by the Vendor Risk team and is not bound by the standard
approval SLA in Section 7.

# 6. Purchase Order Changes
Any change to a released Purchase Order (quantity, price, delivery date) requires the original
approval chain to be re-run. A changed PO does not retain its original approval.

# 7. Service Level Targets
Standard Purchase Orders are targeted for approval within 5 business days of requisition
submission. High-value orders requiring secondary approval are targeted for 10 business days.
These targets are business goals, not contractual guarantees to suppliers.
