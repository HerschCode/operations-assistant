"""
Build instruction-tuning data for tool-selection fine-tuning.

Each example: (system_prompt + user_question) → JSON tool-call.

Data is generated programmatically: template × paraphrase × argument values.
Split is by template_id (not by row) so no template's variants straddle train/test.

Output:
  data/finetune/tool_train.jsonl   -- ShareGPT examples for QLoRA
  data/finetune/tool_test.jsonl    -- held-out templates for evaluation
  data/finetune/tool_stats.json    -- sizes and per-tool counts

Also appends any rows from agent_questions_v2.json that have expected_tools
defined, using the question as-is (these form a small real-trace supplement).

Run: python -m scripts.prepare_tool_selection_data
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
AGENT_QS = ROOT / "data/evaluation/agent_questions_v2.json"
OUT_DIR = ROOT / "data/finetune"
TRAIN_FRAC = 0.80
SEED = 42

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a tool-routing assistant for an operations analytics platform. "
    "Given a user question, output ONLY a JSON object or JSON array specifying "
    "which tool(s) to call and with what arguments. Do not add any explanation.\n\n"
    "Available tools:\n"
    "  get_cycle_time(segment?)          – procurement cycle-time stats\n"
    "  get_bottlenecks(top_n?)           – slowest process stages\n"
    "  get_sla_metrics(segment?)         – SLA breach rate\n"
    "  get_supplier_performance(min_volume?, top_n?)  – supplier scorecard\n"
    "  get_management_report()           – narrative finding+recommendation report\n"
    "  predict_sla_risk(case_id)         – SLA breach probability for one case\n"
    "  get_pipeline_status(limit?)       – recent data-pipeline run status\n"
    "  search_policy_documents(query)    – search procurement policy documents\n"
    "  get_conformance()                 – process conformance rate\n\n"
    "Output format (single tool):\n"
    '  {"tool": "tool_name", "arguments": {"param": "value"}}\n'
    "Output format (multiple tools, in call order):\n"
    '  [{"tool": "tool_name", "arguments": {...}}, ...]'
)

# ── Template definitions ─────────────────────────────────────────────────────
# Each template: id, tool(s), args, question paraphrases.
# Rows generated = len(paraphrases) × len(arg_variants).
# Arg-free templates: arg_variants = [{}]

def _tc(tool: str, args: dict) -> dict:
    return {"tool": tool, "arguments": args}


TEMPLATES: list[dict] = [
    # ── get_cycle_time ────────────────────────────────────────────────────
    {
        "id": "get_cycle_time.overall",
        "calls": [_tc("get_cycle_time", {})],
        "paraphrases": [
            "What is the average procurement cycle time?",
            "How long does procurement typically take?",
            "Give me overall cycle time statistics.",
            "What's the mean, median, and P90 cycle time?",
            "What is our procurement processing time?",
            "How quickly are purchase orders completed on average?",
            "Show me the overall cycle time metrics.",
            "What is the typical end-to-end procurement duration?",
            "How many days does our procurement process take?",
            "Can you give me a summary of cycle time performance?",
            "How efficient is our procurement in terms of time?",
            "What is the P90 procurement cycle time?",
            "On average how long does a purchase order take to process?",
            "Show overall speed metrics for procurement.",
            "Is our procurement cycle time improving?",
            "What's the difference between our mean and median cycle time?",
            "How does our cycle time compare to the P90?",
            "Report on end-to-end procurement timing.",
            "Give me procurement speed KPIs.",
            "What's the fastest and slowest we process orders?",
        ],
    },
    {
        "id": "get_cycle_time.by_category",
        "calls": [_tc("get_cycle_time", {"segment": "category"})],
        "paraphrases": [
            "What is the cycle time broken down by procurement category?",
            "How does cycle time vary across procurement categories?",
            "Show me cycle time per category.",
            "Which procurement category has the longest average cycle time?",
            "Break down cycle time by category.",
            "Give me a per-category view of procurement cycle time.",
            "Compare cycle times across different procurement categories.",
            "How do cycle times differ between categories?",
            "What category is slowest in terms of cycle time?",
            "I need cycle time stats segmented by category.",
            "Which category is fastest?",
            "Segment cycle time by procurement type.",
            "Are some categories processed faster than others?",
            "Show average cycle time for each procurement category.",
        ],
    },
    # ── get_bottlenecks ───────────────────────────────────────────────────
    {
        "id": "get_bottlenecks.default",
        "calls": [_tc("get_bottlenecks", {})],
        "paraphrases": [
            "What are the biggest bottlenecks in our procurement process?",
            "Which process stages are causing the most delays?",
            "Where is procurement slowing down?",
            "Show me the top bottleneck stages.",
            "What are the slowest steps in our procurement workflow?",
            "Identify the main delays in the procurement pipeline.",
            "Which stage takes the longest in our process?",
            "Where are we losing the most time in procurement?",
            "List the process bottlenecks.",
            "What's holding up procurement orders?",
            "Which process stages should we prioritise for improvement?",
            "Analyse our procurement stages for delays.",
            "What are the worst-performing process steps by duration?",
            "Show me the stages ranked by average delay.",
            "What's the biggest source of delay in procurement?",
        ],
    },
    {
        "id": "get_bottlenecks.top3",
        "calls": [_tc("get_bottlenecks", {"top_n": 3})],
        "paraphrases": [
            "What are the top 3 bottlenecks in our process?",
            "Show me the 3 worst bottleneck stages.",
            "Which 3 stages cause the most delay?",
            "Give me the three biggest slowdowns.",
            "List only the 3 most critical bottlenecks.",
            "Tell me the three most problematic process steps.",
            "What are the three biggest procurement delays?",
            "Identify the 3 slowest stages in procurement.",
        ],
    },
    {
        "id": "get_bottlenecks.top5",
        "calls": [_tc("get_bottlenecks", {"top_n": 5})],
        "paraphrases": [
            "What are the top 5 process bottlenecks?",
            "Show me the five slowest stages.",
            "List the 5 biggest delays in procurement.",
            "Which 5 stages are most problematic?",
            "Give me the top 5 bottlenecks.",
            "What are the 5 worst-performing process stages?",
            "I want to see the five stages causing the most delay.",
            "Top 5 bottlenecks in the procurement workflow?",
        ],
    },
    {
        "id": "get_bottlenecks.top20",
        "calls": [_tc("get_bottlenecks", {"top_n": 20})],
        "paraphrases": [
            "List all 20 bottleneck stages.",
            "Give me a comprehensive list of 20 bottleneck stages.",
            "Show the 20 slowest process steps.",
            "I want a full 20-item bottleneck breakdown.",
            "Show me 20 bottlenecks for a complete picture.",
        ],
    },
    {
        "id": "get_bottlenecks.top10",
        "calls": [_tc("get_bottlenecks", {"top_n": 10})],
        "paraphrases": [
            "Show me the top 10 bottleneck stages.",
            "List the 10 slowest process steps.",
            "What are the ten biggest delays in our workflow?",
            "Give me a top-10 list of bottlenecks.",
            "I need the 10 most problematic stages.",
            "Show the 10 worst bottlenecks.",
            "What are the top ten process delays?",
            "List the ten stages with the highest average duration.",
        ],
    },
    # ── get_sla_metrics ───────────────────────────────────────────────────
    {
        "id": "get_sla_metrics.overall",
        "calls": [_tc("get_sla_metrics", {})],
        "paraphrases": [
            "What is our SLA breach rate?",
            "How often are we missing SLA targets?",
            "What percentage of orders breach their SLA?",
            "Give me SLA performance metrics.",
            "How is our SLA compliance?",
            "What fraction of procurement cases are breaching SLA?",
            "Show me the overall SLA breach rate.",
            "Are we meeting our service level agreements?",
            "What's our SLA miss rate?",
            "How many orders are breaching their delivery SLA?",
            "What is our service level agreement compliance rate?",
            "How bad is our SLA problem?",
            "Are we at risk of SLA penalties?",
            "What does our SLA dashboard look like?",
            "How many cases missed the SLA this period?",
        ],
    },
    {
        "id": "get_sla_metrics.by_category",
        "calls": [_tc("get_sla_metrics", {"segment": "category"})],
        "paraphrases": [
            "What is the SLA breach rate by procurement category?",
            "Which category has the worst SLA performance?",
            "Break down SLA compliance by category.",
            "Show me SLA breach rates per category.",
            "How does SLA performance compare across categories?",
            "Which categories are missing their SLA targets?",
            "Give me a per-category SLA breach analysis.",
            "SLA performance breakdown by category.",
            "Compare SLA breach rates across procurement categories.",
            "Which procurement type has the most SLA violations?",
            "SLA breach by category — which is worst?",
            "Show per-category SLA compliance stats.",
            "Which category is most at risk for SLA breaches?",
            "Are all categories equally bad on SLA, or are some worse?",
        ],
    },
    # ── get_supplier_performance ──────────────────────────────────────────
    {
        "id": "get_supplier_performance.default",
        "calls": [_tc("get_supplier_performance", {})],
        "paraphrases": [
            "Which suppliers are performing poorly?",
            "Show me the supplier performance scorecard.",
            "Which suppliers have the highest SLA breach rate?",
            "Give me a supplier performance ranking.",
            "How are our suppliers performing?",
            "Who are our worst-performing suppliers?",
            "Show me supplier SLA breach rates.",
            "What does the supplier performance data show?",
            "Rank suppliers by SLA breach rate.",
            "Which vendors are causing the most SLA issues?",
            "I need a vendor performance review.",
            "Which suppliers should we put on a performance improvement plan?",
            "Show me which vendors to focus on for SLA improvement.",
            "Give me an overview of vendor reliability.",
            "Which suppliers are underperforming against SLA targets?",
        ],
    },
    {
        "id": "get_supplier_performance.high_volume",
        "calls": [_tc("get_supplier_performance", {"min_volume": 20})],
        "paraphrases": [
            "Show me performance for high-volume suppliers only (at least 20 orders).",
            "Which suppliers with 20+ orders have the worst SLA rates?",
            "Supplier performance for vendors with minimum 20 orders.",
            "Filter to suppliers with at least 20 orders and show performance.",
            "Top suppliers by SLA breach rate, minimum volume 20.",
            "Focus on established suppliers — those with at least 20 orders.",
            "Only show suppliers with significant order volume (20 or more).",
            "Supplier scorecard filtered to min_volume of 20.",
        ],
    },
    {
        "id": "get_supplier_performance.top5",
        "calls": [_tc("get_supplier_performance", {"top_n": 5})],
        "paraphrases": [
            "Who are the 5 worst-performing suppliers?",
            "Show me the top 5 suppliers by SLA breach rate.",
            "List the 5 suppliers with the highest breach rates.",
            "Which 5 vendors are performing worst?",
            "Give me only the top 5 underperforming suppliers.",
            "Limit to the worst 5 suppliers.",
            "Just show me 5 suppliers — the worst ones.",
            "Top 5 problematic vendors by SLA breach rate.",
        ],
    },
    # ── get_management_report ─────────────────────────────────────────────
    {
        "id": "get_management_report",
        "calls": [_tc("get_management_report", {})],
        "paraphrases": [
            "Give me the management report.",
            "What does the management summary say?",
            "Show me the current findings and recommendations.",
            "I need the management narrative report.",
            "What are the key findings from the latest report?",
            "Generate the management report.",
            "Summarise the current procurement situation for management.",
            "What does the executive summary look like?",
            "Give me a finding, evidence, impact, and recommendation summary.",
            "Pull up the management report for today.",
            "What are the key recommendations from the analytics system?",
            "I need to brief my manager — what does the report say?",
            "What is the current finding and recommended action?",
            "Get the latest management narrative.",
            "What should management know about current procurement performance?",
        ],
    },
    # ── predict_sla_risk ──────────────────────────────────────────────────
    {
        "id": "predict_sla_risk.C1001",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1001"})],
        "paraphrases": [
            "Is case C1001 at risk of breaching its SLA?",
            "Will C1001 miss its SLA?",
            "Check SLA risk for case C1001.",
            "What's the SLA breach probability for C1001?",
            "Predict whether C1001 will breach its SLA.",
        ],
    },
    {
        "id": "predict_sla_risk.C1023",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1023"})],
        "paraphrases": [
            "Is procurement case C1023 at risk?",
            "What is the SLA risk for C1023?",
            "Will case C1023 breach its SLA?",
            "Predict SLA breach for C1023.",
            "Check if C1023 is going to miss its deadline.",
        ],
    },
    {
        "id": "predict_sla_risk.C1045",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1045"})],
        "paraphrases": [
            "Check SLA risk for procurement order C1045.",
            "Is C1045 going to breach SLA?",
            "Predict breach risk for case C1045.",
            "What's the risk level for C1045?",
            "Will C1045 miss its SLA target?",
        ],
    },
    {
        "id": "predict_sla_risk.C1067",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1067"})],
        "paraphrases": [
            "SLA risk for case C1067?",
            "Is order C1067 at risk of a breach?",
            "Predict whether C1067 will breach.",
            "Check C1067's SLA risk.",
            "How likely is C1067 to breach its SLA?",
        ],
    },
    {
        "id": "predict_sla_risk.C1089",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1089"})],
        "paraphrases": [
            "What is the SLA breach risk for C1089?",
            "Is C1089 at risk?",
            "Predict SLA outcome for case C1089.",
            "Check whether C1089 will miss its SLA.",
            "Risk assessment for procurement case C1089.",
        ],
    },
    {
        "id": "predict_sla_risk.C1100",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1100"})],
        "paraphrases": [
            "SLA prediction for case C1100.",
            "Will order C1100 breach its SLA?",
            "Check C1100 for SLA risk.",
            "Is C1100 likely to be late?",
            "Predict breach probability for C1100.",
        ],
    },
    {
        "id": "predict_sla_risk.C1010",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1010"})],
        "paraphrases": [
            "Is procurement case C1010 on track?",
            "Check SLA risk for C1010.",
            "Will C1010 breach its deadline?",
            "Predict SLA outcome for order C1010.",
            "What's the risk for case C1010?",
        ],
    },
    {
        "id": "predict_sla_risk.C1030",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1030"})],
        "paraphrases": [
            "SLA risk assessment for C1030.",
            "Is C1030 at risk of a delay?",
            "Predict whether C1030 will miss its SLA.",
            "Check C1030's breach probability.",
            "Will case C1030 be on time?",
        ],
    },
    {
        "id": "predict_sla_risk.C1050",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1050"})],
        "paraphrases": [
            "What is the SLA risk for C1050?",
            "Is C1050 going to breach?",
            "Check if case C1050 is at risk.",
            "Predict breach for procurement order C1050.",
            "SLA prediction for C1050.",
        ],
    },
    {
        "id": "predict_sla_risk.C1070",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1070"})],
        "paraphrases": [
            "Check SLA risk for case C1070.",
            "Is order C1070 likely to breach?",
            "Predict SLA for C1070.",
            "What's the breach probability for C1070?",
            "Is C1070 at risk of missing its deadline?",
        ],
    },
    {
        "id": "predict_sla_risk.C1090",
        "calls": [_tc("predict_sla_risk", {"case_id": "C1090"})],
        "paraphrases": [
            "SLA risk for case C1090.",
            "Will C1090 breach its SLA target?",
            "Predict breach for C1090.",
            "Is C1090 on schedule?",
            "Check procurement case C1090 for SLA risk.",
        ],
    },
    # ── get_pipeline_status ───────────────────────────────────────────────
    {
        "id": "get_pipeline_status.default",
        "calls": [_tc("get_pipeline_status", {})],
        "paraphrases": [
            "What is the status of the data pipeline?",
            "When did the last pipeline run complete?",
            "Is the data pipeline up to date?",
            "Show me recent pipeline runs.",
            "Has the data been refreshed recently?",
            "Are there any pipeline failures I should know about?",
            "What's the current state of the data ingestion pipeline?",
            "Check if the pipeline ran successfully.",
            "Is the underlying data fresh?",
            "Show me the pipeline run history.",
            "Did the last ETL job succeed?",
            "When was the data warehouse last updated?",
            "Are there any pipeline errors I should be aware of?",
            "How recently was the metrics data refreshed?",
        ],
    },
    {
        "id": "get_pipeline_status.limit3",
        "calls": [_tc("get_pipeline_status", {"limit": 3})],
        "paraphrases": [
            "Show me the 3 most recent pipeline runs.",
            "What were the last 3 data pipeline executions?",
            "Pipeline status for the 3 most recent runs.",
            "Check the last 3 pipeline jobs.",
            "Give me the 3 latest pipeline run results.",
        ],
    },
    {
        "id": "get_pipeline_status.recent",
        "calls": [_tc("get_pipeline_status", {"limit": 5})],
        "paraphrases": [
            "Show me the last 5 pipeline runs.",
            "What happened in the 5 most recent pipeline executions?",
            "Give me the status of the last 5 data pipeline runs.",
            "Check the 5 most recent pipeline runs.",
            "Show pipeline history for the last 5 runs.",
            "I need the 5 most recent pipeline statuses.",
            "Last 5 data pipeline runs, please.",
            "Display status for the 5 most recent ingestion runs.",
        ],
    },
    # ── search_policy_documents ───────────────────────────────────────────
    {
        "id": "search_policy_documents.approval",
        "calls": [_tc("search_policy_documents", {"query": "approval thresholds"})],
        "paraphrases": [
            "What are the approval thresholds for purchase orders?",
            "At what value does a purchase order require additional approval?",
            "Who needs to approve procurement orders and at what amount?",
            "What is the approval process for large purchases?",
            "What are the different approval levels in procurement?",
            "What dollar amount triggers a second approval level?",
            "Do all purchase orders need manager sign-off?",
            "What is the single-approval purchase limit?",
        ],
    },
    {
        "id": "search_policy_documents.sla_breach",
        "calls": [_tc("search_policy_documents", {"query": "SLA breach penalties escalation"})],
        "paraphrases": [
            "What happens when an SLA is breached?",
            "What are the penalties for SLA violations?",
            "How are SLA breaches escalated?",
            "What is the policy for handling SLA breaches?",
            "What are the consequences of missing SLA targets?",
            "Is there a financial penalty for SLA breaches?",
            "How quickly must an SLA breach be reported?",
        ],
    },
    {
        "id": "search_policy_documents.escalation",
        "calls": [_tc("search_policy_documents", {"query": "escalation procedure"})],
        "paraphrases": [
            "What is the escalation procedure for procurement issues?",
            "How do I escalate a procurement problem?",
            "Who do I escalate to when something goes wrong?",
            "What are the escalation paths for procurement delays?",
            "Describe the escalation process for unresolved issues.",
        ],
    },
    {
        "id": "search_policy_documents.vendor_onboarding",
        "calls": [_tc("search_policy_documents", {"query": "vendor onboarding requirements"})],
        "paraphrases": [
            "What are the vendor onboarding requirements?",
            "How do we onboard new suppliers?",
            "What does a supplier need to provide to be onboarded?",
            "What is the vendor registration process?",
            "What documents are required for supplier onboarding?",
        ],
    },
    {
        "id": "search_policy_documents.contract_renewal",
        "calls": [_tc("search_policy_documents", {"query": "contract renewal policy"})],
        "paraphrases": [
            "What is the contract renewal policy?",
            "How are contracts renewed with existing suppliers?",
            "What is the process for renewing supplier contracts?",
            "When should contracts be renewed and who approves them?",
            "What are the rules around contract renewal?",
        ],
    },
    {
        "id": "search_policy_documents.exception",
        "calls": [_tc("search_policy_documents", {"query": "exception handling waiver"})],
        "paraphrases": [
            "How do we handle procurement exceptions?",
            "What is the process for requesting a procurement waiver?",
            "When can exceptions to procurement policy be approved?",
            "Who can grant procurement exceptions?",
            "What is the exception handling procedure?",
        ],
    },
    {
        "id": "search_policy_documents.consignment",
        "calls": [_tc("search_policy_documents", {"query": "consignment stock"})],
        "paraphrases": [
            "What is the policy on consignment stock?",
            "How do we handle consignment inventory?",
            "What are the rules for consignment goods?",
            "Is consignment stock allowed under procurement policy?",
            "What does the policy say about consignment arrangements?",
        ],
    },
    {
        "id": "search_policy_documents.three_way_match",
        "calls": [_tc("search_policy_documents", {"query": "three-way match invoice"})],
        "paraphrases": [
            "What is the three-way match process?",
            "How does invoice matching work?",
            "What does three-way match mean in procurement?",
            "Explain the PO, GR, invoice matching requirement.",
            "What are the invoice verification procedures?",
        ],
    },
    {
        "id": "search_policy_documents.emergency_purchase",
        "calls": [_tc("search_policy_documents", {"query": "emergency purchase urgent procurement"})],
        "paraphrases": [
            "What is the process for emergency purchases?",
            "Can we bypass the normal approval process for urgent orders?",
            "How do we handle urgent procurement requests?",
            "What is the policy for emergency or expedited procurement?",
            "Is there a fast-track process for emergency purchases?",
        ],
    },
    {
        "id": "search_policy_documents.sole_source",
        "calls": [_tc("search_policy_documents", {"query": "sole source single supplier justification"})],
        "paraphrases": [
            "What is the sole-source procurement policy?",
            "When can we buy from a single supplier without competitive bidding?",
            "How do we justify sole-source procurement?",
            "What are the rules for single-supplier orders?",
            "Can we skip competitive bidding for sole-source suppliers?",
        ],
    },
    {
        "id": "search_policy_documents.payment_terms",
        "calls": [_tc("search_policy_documents", {"query": "payment terms invoice due date"})],
        "paraphrases": [
            "What are the standard payment terms with suppliers?",
            "How many days do we have to pay invoices?",
            "What is the policy on payment deadlines?",
            "What payment terms does procurement use?",
            "What are our standard invoice payment terms?",
        ],
    },
    {
        "id": "search_policy_documents.preferred_supplier",
        "calls": [_tc("search_policy_documents", {"query": "preferred supplier approved vendor list"})],
        "paraphrases": [
            "What is the preferred supplier program?",
            "How do suppliers get on the approved vendor list?",
            "What are the criteria for preferred supplier status?",
            "How do we manage the approved vendor list?",
            "What is the policy on using preferred suppliers?",
        ],
    },
    # ── get_conformance ───────────────────────────────────────────────────
    {
        "id": "get_conformance",
        "calls": [_tc("get_conformance", {})],
        "paraphrases": [
            "What is our process conformance rate?",
            "How many orders follow the standard procurement process?",
            "What fraction of cases conform to the expected workflow?",
            "Are procurement cases following the correct process steps?",
            "What is our workflow compliance rate?",
            "How many orders deviate from the standard procurement flow?",
            "Show me process conformance metrics.",
            "What percentage of orders follow the required steps in order?",
            "Is our process being followed correctly?",
            "What is the conformance rate for the procurement workflow?",
            "How compliant is our procurement process?",
            "Are people following the correct procurement steps?",
            "What proportion of orders go through the expected process flow?",
            "Is there a lot of process deviation in procurement?",
            "How many cases skip required steps in the procurement process?",
        ],
    },
    # ── Multi-tool templates ──────────────────────────────────────────────
    {
        "id": "multi.cycle_sla",
        "calls": [_tc("get_cycle_time", {}), _tc("get_sla_metrics", {})],
        "paraphrases": [
            "What is our cycle time and SLA breach rate?",
            "Give me both cycle time and SLA metrics.",
            "How long does procurement take and how often do we breach SLA?",
            "I need cycle time stats and SLA performance numbers.",
            "Show me procurement speed and SLA compliance together.",
            "What are our key performance indicators: cycle time and SLA?",
            "Summarise cycle time and SLA breach rate.",
            "How is our procurement performing on time and SLA dimensions?",
        ],
    },
    {
        "id": "multi.cycle_sla_by_cat",
        "calls": [
            _tc("get_cycle_time", {"segment": "category"}),
            _tc("get_sla_metrics", {"segment": "category"}),
        ],
        "paraphrases": [
            "Break down both cycle time and SLA breach rate by category.",
            "Which procurement category has the worst performance overall?",
            "Compare categories on both speed and SLA compliance.",
            "Give me cycle time and SLA metrics per procurement category.",
            "Show a per-category breakdown of cycle time and SLA.",
        ],
    },
    {
        "id": "multi.bottleneck_policy",
        "calls": [
            _tc("get_bottlenecks", {}),
            _tc("search_policy_documents", {"query": "escalation procedure"}),
        ],
        "paraphrases": [
            "Where are the bottlenecks and what does policy say about escalation?",
            "Show me bottlenecks and look up the escalation procedure.",
            "What are our worst delays and what's the escalation policy?",
            "Get bottleneck data and search for escalation procedures.",
            "Identify delays and check the escalation rules.",
        ],
    },
    {
        "id": "multi.cycle_policy",
        "calls": [
            _tc("get_cycle_time", {"segment": "category"}),
            _tc("search_policy_documents", {"query": "SLA targets by category"}),
        ],
        "paraphrases": [
            "What's the cycle time by category and what are the policy SLA targets?",
            "Show cycle time per category and look up SLA targets from policy.",
            "Get category cycle times and the SLA requirements from the policy doc.",
            "Compare per-category cycle time against the SLA policy.",
            "Which category is slowest and what does policy say about their SLA?",
        ],
    },
    {
        "id": "multi.sla_pipeline",
        "calls": [_tc("get_sla_metrics", {}), _tc("get_pipeline_status", {})],
        "paraphrases": [
            "Are the SLA numbers based on fresh data? Show SLA metrics and pipeline status.",
            "Check SLA breach rate and confirm the pipeline is up to date.",
            "Give me SLA metrics and tell me when the data was last refreshed.",
            "SLA performance and pipeline status please.",
            "I want to see SLA numbers but first confirm the data pipeline ran recently.",
        ],
    },
    {
        "id": "multi.supplier_policy",
        "calls": [
            _tc("get_supplier_performance", {}),
            _tc("search_policy_documents", {"query": "supplier performance requirements"}),
        ],
        "paraphrases": [
            "Which suppliers are underperforming and what does policy say about it?",
            "Show supplier scorecard and look up performance policy.",
            "Get supplier performance data and the relevant policy sections.",
            "List worst suppliers and check what the policy requires of them.",
            "Show me the supplier leaderboard and search for performance standards.",
        ],
    },
    {
        "id": "multi.report_conformance",
        "calls": [_tc("get_management_report", {}), _tc("get_conformance", {})],
        "paraphrases": [
            "Generate the management report and show conformance rate.",
            "I need the management summary and process conformance stats.",
            "Give me the management narrative and the conformance metric.",
            "Pull the management report alongside conformance data.",
            "Management report plus process conformance, please.",
        ],
    },
    {
        "id": "multi.three.cycle_sla_bottleneck",
        "calls": [
            _tc("get_cycle_time", {}),
            _tc("get_sla_metrics", {}),
            _tc("get_bottlenecks", {}),
        ],
        "paraphrases": [
            "Give me cycle time, SLA breach rate, and the main bottlenecks.",
            "I need a full performance overview: cycle time, SLA, and bottlenecks.",
            "Show me cycle time stats, SLA metrics, and where the process is slow.",
            "What are our cycle time, SLA performance, and process bottlenecks?",
            "Three things: overall cycle time, SLA breach rate, and top bottlenecks.",
            "Give me the complete picture — cycle time, SLA, and bottleneck data.",
        ],
    },
    {
        "id": "multi.three.supplier_policy_sla",
        "calls": [
            _tc("get_supplier_performance", {}),
            _tc("get_sla_metrics", {}),
            _tc("search_policy_documents", {"query": "supplier performance requirements"}),
        ],
        "paraphrases": [
            "Show supplier performance, SLA breach rates, and look up the performance policy.",
            "I need supplier data, overall SLA numbers, and the relevant policy sections.",
            "Get me the supplier scorecard, SLA metrics, and policy on supplier standards.",
            "Combine supplier performance, SLA stats, and policy doc search for supplier requirements.",
            "What are the suppliers doing, what's the SLA rate, and what does policy require?",
            "Give me supplier performance and SLA data, then search the policy for requirements.",
        ],
    },
    {
        "id": "multi.three.pipeline_cycle_sla",
        "calls": [
            _tc("get_pipeline_status", {}),
            _tc("get_cycle_time", {}),
            _tc("get_sla_metrics", {}),
        ],
        "paraphrases": [
            "Confirm the data is fresh, then show me cycle time and SLA metrics.",
            "Check pipeline status first, then give me cycle time and SLA numbers.",
            "Is the data up to date? If so, show cycle time and SLA breach rates.",
            "Pipeline status plus cycle time and SLA performance please.",
            "I want to verify data freshness, then see cycle time and SLA metrics.",
            "Pipeline status, cycle time, and SLA — give me all three.",
        ],
    },
    {
        "id": "multi.cycle_conformance",
        "calls": [_tc("get_cycle_time", {}), _tc("get_conformance", {})],
        "paraphrases": [
            "What is our cycle time and process conformance rate?",
            "Show me both cycle time and how well the process is being followed.",
            "I want cycle time stats and the conformance rate together.",
            "How fast is procurement and how often is the process followed correctly?",
            "Give me cycle time and process compliance metrics.",
            "Cycle time and conformance rate, please.",
            "Are we fast and are we following the right steps? Show me both.",
            "Get me speed metrics and process compliance at the same time.",
        ],
    },
    {
        "id": "multi.bottleneck_conformance",
        "calls": [_tc("get_bottlenecks", {}), _tc("get_conformance", {})],
        "paraphrases": [
            "What are the bottlenecks and what's the conformance rate?",
            "Show me process delays and how well the workflow is being followed.",
            "I need bottleneck data and the process conformance metric.",
            "Which stages are slow and what fraction of cases follow the standard process?",
            "Get me bottleneck rankings and the overall conformance rate.",
            "Bottlenecks and conformance together, please.",
            "Are our slow stages also where people skip steps?",
            "Show me where delays happen and how compliant the process is.",
        ],
    },
    {
        "id": "multi.sla_conformance",
        "calls": [_tc("get_sla_metrics", {}), _tc("get_conformance", {})],
        "paraphrases": [
            "What is our SLA breach rate and process conformance?",
            "How compliant are we on both SLA targets and process steps?",
            "SLA performance and workflow conformance, please.",
            "Show me SLA metrics alongside process conformance.",
            "Give me SLA and conformance numbers together.",
            "Are SLA breaches related to process non-conformance? Show me both.",
            "I want to understand SLA compliance and process compliance.",
            "SLA breach rate and conformance rate — both please.",
        ],
    },
    {
        "id": "multi.supplier_cycle",
        "calls": [_tc("get_supplier_performance", {}), _tc("get_cycle_time", {"segment": "category"})],
        "paraphrases": [
            "Show me supplier performance and cycle time by category.",
            "Which suppliers are slow and which categories take longest?",
            "Combine supplier scorecard with per-category cycle time.",
            "I need supplier performance data and cycle time broken down by category.",
            "Get supplier rankings and category-level cycle time.",
            "Supplier performance plus category cycle time breakdown.",
        ],
    },
    {
        "id": "multi.pipeline_bottleneck",
        "calls": [_tc("get_pipeline_status", {}), _tc("get_bottlenecks", {})],
        "paraphrases": [
            "Is the data current? Also show me the bottlenecks.",
            "Check pipeline status and give me the bottleneck analysis.",
            "Show pipeline health and process bottlenecks.",
            "I want to see if the data is fresh and where the delays are.",
            "Pipeline status and bottleneck report, please.",
            "Confirm data freshness, then show bottlenecks.",
        ],
    },
    # ── Extra single-tool variants ────────────────────────────────────────
    {
        "id": "get_supplier_performance.min10",
        "calls": [_tc("get_supplier_performance", {"min_volume": 10})],
        "paraphrases": [
            "Show supplier performance for vendors with at least 10 orders.",
            "Filter suppliers to those with 10 or more orders.",
            "Supplier scorecard, minimum 10 orders.",
            "Which suppliers with 10+ orders are underperforming?",
            "Performance data for suppliers with at least 10 orders.",
            "Vendor performance filtered to 10-order minimum.",
        ],
    },
    {
        "id": "get_supplier_performance.top10",
        "calls": [_tc("get_supplier_performance", {"top_n": 10})],
        "paraphrases": [
            "Show me the 10 worst-performing suppliers.",
            "Top 10 suppliers by SLA breach rate.",
            "List the 10 vendors with the most SLA issues.",
            "Give me the worst 10 suppliers.",
            "I need the top 10 underperforming vendors.",
            "Which 10 suppliers should we review first?",
        ],
    },
    {
        "id": "search_policy_documents.dispute",
        "calls": [_tc("search_policy_documents", {"query": "supplier dispute resolution"})],
        "paraphrases": [
            "What is the process for resolving supplier disputes?",
            "How do we handle a dispute with a supplier?",
            "What does the policy say about supplier conflict resolution?",
            "How are supplier disagreements escalated and resolved?",
            "What is the supplier dispute resolution procedure?",
            "Who handles supplier disputes and how?",
        ],
    },
    {
        "id": "search_policy_documents.audit",
        "calls": [_tc("search_policy_documents", {"query": "procurement audit requirements"})],
        "paraphrases": [
            "What are the audit requirements for procurement?",
            "How often must procurement records be audited?",
            "What documentation is required for a procurement audit?",
            "What does the policy say about procurement audits?",
            "How do we prepare for a procurement audit?",
            "What records need to be kept for audit purposes?",
        ],
    },
]


def _answer_json(calls: list[dict]) -> str:
    if len(calls) == 1:
        return json.dumps(calls[0], separators=(",", ":"))
    return json.dumps(calls, separators=(",", ":"))


def _build_examples(templates: list[dict]) -> list[dict]:
    examples = []
    for tmpl in templates:
        answer = _answer_json(tmpl["calls"])
        for q in tmpl["paraphrases"]:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer},
                    ],
                    "meta": {
                        "template_id": tmpl["id"],
                        "tools": [c["tool"] for c in tmpl["calls"]],
                    },
                }
            )
    return examples


def _split_by_template(
    examples: list[dict], train_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by template_id so no template straddles train/test."""
    from collections import defaultdict

    by_tmpl: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_tmpl[ex["meta"]["template_id"]].append(ex)

    tmpl_ids = sorted(by_tmpl)
    rng = random.Random(seed)
    rng.shuffle(tmpl_ids)
    n_train = max(1, round(len(tmpl_ids) * train_frac))
    train_ids = set(tmpl_ids[:n_train])

    train = [ex for ex in examples if ex["meta"]["template_id"] in train_ids]
    test = [ex for ex in examples if ex["meta"]["template_id"] not in train_ids]
    return train, test


def _check_leakage(train: list[dict], test: list[dict]) -> None:
    train_tmpl = {ex["meta"]["template_id"] for ex in train}
    test_tmpl = {ex["meta"]["template_id"] for ex in test}
    overlap = train_tmpl & test_tmpl
    if overlap:
        raise ValueError(f"Template leakage: {overlap} appear in both train and test")


def _add_agent_traces(examples: list[dict]) -> list[dict]:
    """Append real agent-eval questions that have expected_tools defined."""
    try:
        qs = json.loads(AGENT_QS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return examples

    added = []
    for q in qs:
        tools = q.get("expected_tools") or []
        if not tools:
            continue
        args_map = q.get("expected_args") or {}
        calls = [
            {"tool": t, "arguments": args_map.get(t, {})}
            for t in tools
        ]
        answer = _answer_json(calls)
        added.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "template_id": f"agent_trace.{q['id']}",
                    "tools": tools,
                    "source": "agent_questions_v2",
                },
            }
        )
    return examples + added


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = _build_examples(TEMPLATES)
    examples = _add_agent_traces(examples)

    train, test = _split_by_template(examples, TRAIN_FRAC, SEED)
    _check_leakage(train, test)

    # Shuffle within each split
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(test)

    train_path = OUT_DIR / "tool_train.jsonl"
    test_path = OUT_DIR / "tool_test.jsonl"
    stats_path = OUT_DIR / "tool_stats.json"

    train_path.write_text(
        "\n".join(json.dumps(e) for e in train) + "\n", encoding="utf-8"
    )
    test_path.write_text(
        "\n".join(json.dumps(e) for e in test) + "\n", encoding="utf-8"
    )

    from collections import Counter
    tool_counts = Counter(t for ex in train + test for t in ex["meta"]["tools"])
    train_tmpls = {ex["meta"]["template_id"] for ex in train}
    test_tmpls = {ex["meta"]["template_id"] for ex in test}

    stats = {
        "n_templates": len(train_tmpls) + len(test_tmpls),
        "n_train": len(train),
        "n_train_templates": len(train_tmpls),
        "n_test": len(test),
        "n_test_templates": len(test_tmpls),
        "train_frac_templates": round(len(train_tmpls) / (len(train_tmpls) + len(test_tmpls)), 3),
        "tool_counts": dict(tool_counts.most_common()),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Tool-selection dataset built:")
    print(f"  train: {stats['n_train']} examples  ({stats['n_train_templates']} templates)")
    print(f"  test:  {stats['n_test']} examples  ({stats['n_test_templates']} templates)")
    print(f"  total: {stats['n_train'] + stats['n_test']} examples")
    print(f"\nTool distribution (train+test):")
    for t, n in tool_counts.most_common():
        print(f"  {t:<35} {n}")
    print(f"\nWrote {train_path.relative_to(ROOT)}")
    print(f"      {test_path.relative_to(ROOT)}")
    print(f"      {stats_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
