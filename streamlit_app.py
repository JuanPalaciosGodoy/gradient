import uuid

import streamlit as st

from app.audit.report_builder import build_report
from app.audit.task_classifier import classify_task
from app.database import init_db, save_audit_run, save_report, save_usage_records
from app.ingestion.csv_loader import load_csv
from app.reports.markdown import render_markdown_report

init_db()

st.set_page_config(page_title="Gradient — AI Procurement Audit", layout="wide")
st.title("Gradient")
st.caption("AI Procurement Audit — Upload your usage data. Get an executive audit.")

uploaded_file = st.file_uploader("Upload AI usage CSV", type="csv")

if uploaded_file:
    content = uploaded_file.read()
    try:
        records = load_csv(content)
    except ValueError as e:
        st.error(f"CSV validation failed: {e}")
        st.stop()

    st.success(f"Loaded {len(records):,} records.")

    if st.button("Generate Audit Report", type="primary"):
        with st.spinner("Analyzing usage data…"):
            run_id = str(uuid.uuid4())
            save_audit_run(run_id=run_id, file_name=uploaded_file.name, record_count=len(records))
            records_dicts = []
            for r in records:
                d = r.model_dump(mode="json")
                if d.get("task_type") is None:
                    d["task_type"] = classify_task(r.prompt).value
                records_dicts.append(d)
            save_usage_records(run_id, records_dicts)
            report = build_report(run_id, records)
            save_report(report)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Annual Spend", f"${report.current_annual_spend:,.0f}")
        col2.metric("Potential Annual Savings", f"${report.potential_annual_savings:,.0f}")
        col3.metric("Confidence", f"{report.confidence_score:.0%}")

        st.subheader("Executive Summary")
        st.write(report.executive_summary)

        if report.top_opportunities:
            st.subheader("Top Opportunities")
            for i, opp in enumerate(report.top_opportunities, 1):
                label = opp.task_type.replace("_", " ").title()
                with st.expander(f"{i}. {label} — ${opp.estimated_annual_savings:,.0f}/yr potential savings"):
                    st.write(f"**Recommendation:** {opp.recommended_action}")
                    st.write(f"**Rationale:** {opp.rationale}")
                    st.write(f"**Confidence:** {opp.confidence:.0%}")

        if report.risk_notes:
            st.subheader("Risk Notes")
            for note in report.risk_notes:
                severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(note.severity, "⚪")
                st.write(f"{severity_color} **{note.category.replace('_', ' ').title()}:** {note.description}")

        markdown = render_markdown_report(report)
        st.download_button(
            label="Download Markdown Report",
            data=markdown,
            file_name=f"gradient_audit_{run_id[:8]}.md",
            mime="text/markdown",
        )
