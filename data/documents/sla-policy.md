---
title: SLA Policy
version: "1.4"
effective_date: 2024-02-01
department: Operations
---

# 1. Purpose
Defines the internal service-level targets for procurement cycle time by spend category.

# 2. Targets by Category
| Category | Target (business days) |
|---|---|
| 3-way match | 10 |
| Consignment | 14 |
| Standard | 10 (default) |

These map directly to `config/sla.yaml` in the operations-performance project -- if this table
and that file ever disagree, the file is authoritative for what the system actually enforces;
this document should be updated to match, not the other way around.

# 3. What Counts Toward Cycle Time
Cycle time is measured from Purchase Requisition creation to final Payment, inclusive of any
Manual Credit Review or PO Change Approval time (Section 5 and Section 6 of the Procurement
Policy). It is not paused or excluded during those steps, even though they are outside the
standard approval SLA.

# 4. Breach Reporting
Any case whose cycle time exceeds its category's target is reported as an SLA breach in the
weekly operations management report. A single breach does not trigger automatic escalation --
see the Escalation Procedure for when escalation is required.

# 5. Review Cadence
SLA targets are reviewed quarterly by Operations leadership and may be revised based on observed
process performance and business priorities.
