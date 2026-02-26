"""
DriveMindr Review Dashboard — Streamlit-based UI for reviewing AI classifications.

Three views:
  1. Drive Overview — storage usage, top-20 largest, extension breakdown
  2. Action Review — tabs for each classification, approve/reject per item & batch
  3. Execution — approved action plan summary (read-only, no execution in Phase 3)

Reads from SQLite. Writes user decisions back. Nothing is executed from here.
Launch: streamlit run drivemindr/dashboard.py -- --db drivemindr.db
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure drivemindr is importable when launched via `streamlit run`
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from drivemindr.database import Database  # noqa: E402
from drivemindr.utils import format_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DriveMindr",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Database connection (cached per session)
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Resolve database path from CLI args or default."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--db" and i + 1 < len(args):
            return args[i + 1]
    return "drivemindr.db"


@st.cache_resource
def get_db() -> Database:
    """Open a shared DB connection for the Streamlit session."""
    db = Database(_get_db_path())
    db.connect()
    return db


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("DriveMindr")
st.sidebar.caption("AI-Powered Storage Manager")
view = st.sidebar.radio(
    "View",
    ["Drive Overview", "Action Review", "Execution Plan"],
    label_visibility="collapsed",
)

db = get_db()


# ===================================================================
# VIEW 1: Drive Overview
# ===================================================================
def render_drive_overview() -> None:
    st.header("Drive Overview")

    # Key metrics row
    file_count = db.file_count()
    total_size = db.total_size()
    review_stats = db.get_review_stats()
    recovery = db.get_space_recovery_estimate()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Files", f"{file_count:,}")
    col2.metric("Total Size", format_bytes(total_size))
    col3.metric("Classified", f"{review_stats['classified']:,}")
    col4.metric(
        "Recoverable",
        format_bytes(sum(recovery.values())),
    )

    st.divider()

    # Classification summary
    summary = db.get_classification_summary()
    if summary:
        st.subheader("Classification Summary")
        cols = st.columns(min(len(summary), 6))
        for i, (action, data) in enumerate(summary.items()):
            with cols[i % len(cols)]:
                st.metric(
                    action.replace("_", " ").title(),
                    f"{data['count']:,} files",
                    format_bytes(data["bytes"]),
                )

        # Bar chart of file counts by action
        chart_data = {
            action: data["count"] for action, data in summary.items()
        }
        st.bar_chart(chart_data)
    else:
        st.info(
            "No classifications yet. Run `drivemindr classify` first.",
            icon="ℹ️",
        )

    st.divider()

    # Top 20 largest files
    left, right = st.columns(2)

    with left:
        st.subheader("Top 20 Largest Files")
        largest = db.get_top_largest(20)
        if largest:
            rows = []
            for f in largest:
                rows.append({
                    "File": f["name"],
                    "Size": format_bytes(f["size_bytes"]),
                    "Path": f["path"],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No files scanned yet.")

    # Extension breakdown
    with right:
        st.subheader("Top Extensions by Size")
        ext_data = db.get_extension_breakdown()
        if ext_data:
            rows = []
            for e in ext_data:
                rows.append({
                    "Extension": e["extension"],
                    "Files": e["file_count"],
                    "Total Size": format_bytes(e["total_bytes"]),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # Top directories
    st.subheader("Largest Directories")
    dir_sizes = db.get_dir_sizes(20)
    if dir_sizes:
        rows = []
        for d in dir_sizes:
            rows.append({
                "Directory": d["path"],
                "Size": format_bytes(d["total_bytes"]),
                "Files": d["file_count"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Space recovery estimate
    if recovery:
        st.subheader("Estimated Space Recovery")
        for action, bytes_val in recovery.items():
            label = action.replace("_", " ").title()
            st.write(f"**{label}:** {format_bytes(bytes_val)}")


# ===================================================================
# VIEW 2: Action Review
# ===================================================================
ACTION_TABS = ["DELETE_JUNK", "DELETE_UNUSED", "MOVE_DATA", "MOVE_APP", "ARCHIVE", "KEEP"]
ACTION_COLORS = {
    "DELETE_JUNK": "🔴",
    "DELETE_UNUSED": "🟠",
    "MOVE_DATA": "🔵",
    "MOVE_APP": "🟣",
    "ARCHIVE": "🟡",
    "KEEP": "🟢",
}


def _render_file_table(files: list, action: str) -> None:
    """Render a reviewable table of files for a given action category."""
    if not files:
        st.info(f"No files classified as {action.replace('_', ' ').title()}.")
        return

    st.caption(f"{len(files)} files")

    # Batch actions
    batch_col1, batch_col2, batch_col3 = st.columns([1, 1, 4])
    batch_key = f"batch_{action}"
    with batch_col1:
        if st.button(f"Approve All", key=f"approve_all_{action}", type="primary"):
            file_ids = [f["id"] for f in files]
            db.save_batch_decisions(file_ids, "APPROVE")
            st.success(f"Approved {len(file_ids)} files.")
            st.rerun()
    with batch_col2:
        if st.button(f"Reject All", key=f"reject_all_{action}"):
            file_ids = [f["id"] for f in files]
            db.save_batch_decisions(file_ids, "REJECT")
            st.success(f"Rejected {len(file_ids)} files.")
            st.rerun()

    st.divider()

    # Individual file review
    for f in files:
        file_id = f["id"]
        decision = f["decision"] if "decision" in f.keys() else None
        confidence = f["confidence"]

        # Status badge
        if decision:
            badge = {"APPROVE": "✅", "REJECT": "❌", "PROTECT": "🛡️"}.get(decision, "⏳")
            status = f"{badge} {decision}"
        elif f["overridden"]:
            status = "🛡️ SAFETY OVERRIDE"
        else:
            status = "⏳ Pending"

        # Confidence indicator
        if confidence >= 0.85:
            conf_color = "🟢"
        elif confidence >= 0.7:
            conf_color = "🟡"
        else:
            conf_color = "🔴"

        with st.expander(
            f"{f['name']}  —  {format_bytes(f['size_bytes'])}  |  "
            f"Confidence: {conf_color} {confidence:.0%}  |  {status}",
            expanded=False,
        ):
            info_col, action_col = st.columns([3, 1])

            with info_col:
                st.text(f"Path:     {f['path']}")
                st.text(f"Size:     {format_bytes(f['size_bytes'])}")
                st.text(f"Modified: {f['modified']}")
                st.text(f"Accessed: {f['accessed']}")
                st.text(f"AI Reason: {f['reason']}")
                if f["overridden"]:
                    st.warning(f"Safety override: {f['override_reason']}")

            with action_col:
                st.write("**Decision:**")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Approve", key=f"approve_{file_id}", type="primary"):
                        db.save_user_decision(file_id, "APPROVE")
                        st.rerun()
                with c2:
                    if st.button("Reject", key=f"reject_{file_id}"):
                        db.save_user_decision(file_id, "REJECT")
                        st.rerun()

                if st.button("Protect", key=f"protect_{file_id}"):
                    db.save_user_decision(file_id, "PROTECT", "KEEP")
                    st.rerun()

                # Change action dropdown
                new_action = st.selectbox(
                    "Change to:",
                    [a for a in ACTION_TABS if a != action],
                    key=f"change_{file_id}",
                    label_visibility="collapsed",
                )
                if st.button("Change", key=f"do_change_{file_id}"):
                    db.save_user_decision(file_id, "CHANGE", new_action)
                    st.rerun()


def render_action_review() -> None:
    st.header("Action Review")

    stats = db.get_review_stats()

    # Progress bar
    if stats["classified"] > 0:
        progress = stats["reviewed"] / stats["classified"]
        st.progress(progress, text=f"Reviewed {stats['reviewed']:,} / {stats['classified']:,} files ({progress:.0%})")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Classified", f"{stats['classified']:,}")
    metric_cols[1].metric("Reviewed", f"{stats['reviewed']:,}")
    metric_cols[2].metric("Approved", f"{stats['approved']:,}")
    metric_cols[3].metric("Rejected", f"{stats['rejected']:,}")
    metric_cols[4].metric("Pending", f"{stats['pending']:,}")

    st.divider()

    # Filter controls
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        show_reviewed = st.checkbox("Show already reviewed", value=False)
    with filter_col2:
        confidence_filter = st.slider(
            "Min confidence", 0.0, 1.0, 0.0, 0.05,
            help="Only show files with confidence >= this value",
        )

    # Tabs for each action type
    tab_labels = [f"{ACTION_COLORS.get(a, '')} {a.replace('_', ' ').title()}" for a in ACTION_TABS]
    tabs = st.tabs(tab_labels)

    for tab, action in zip(tabs, ACTION_TABS):
        with tab:
            files = db.get_files_by_action(action)
            # Apply filters
            filtered = []
            for f in files:
                if not show_reviewed and f["decision"] is not None:
                    continue
                if f["confidence"] < confidence_filter:
                    continue
                filtered.append(f)
            _render_file_table(filtered, action)


# ===================================================================
# VIEW 3: Execution Plan
# ===================================================================
def render_execution_plan() -> None:
    st.header("Execution Plan")
    st.info(
        "This view shows the approved action plan. "
        "Execution will be available in Phase 4.",
        icon="ℹ️",
    )

    approved = db.get_approved_actions()

    if not approved:
        st.warning("No actions approved yet. Review files in the Action Review tab first.")
        return

    # Group by action
    by_action: dict[str, list] = {}
    total_bytes = 0
    for f in approved:
        action = f["final_action"]
        by_action.setdefault(action, []).append(f)
        if action.startswith("DELETE"):
            total_bytes += f["size_bytes"]

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Approved", f"{len(approved):,} files")
    col2.metric("Actions", f"{len(by_action)} categories")
    col3.metric("Space to Free", format_bytes(total_bytes))

    st.divider()

    # Breakdown by action
    for action, files in sorted(by_action.items()):
        emoji = ACTION_COLORS.get(action, "⚪")
        action_bytes = sum(f["size_bytes"] for f in files)
        st.subheader(f"{emoji} {action.replace('_', ' ').title()} — {len(files)} files ({format_bytes(action_bytes)})")

        rows = []
        for f in files:
            rows.append({
                "File": f["name"],
                "Size": format_bytes(f["size_bytes"]),
                "Path": f["path"],
                "Confidence": f"{f['confidence']:.0%}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # Dry run summary
    st.subheader("Action Plan Summary")
    for action, files in sorted(by_action.items()):
        action_bytes = sum(f["size_bytes"] for f in files)
        st.write(f"- **{action.replace('_', ' ').title()}**: {len(files)} files ({format_bytes(action_bytes)})")

    st.caption(
        "This plan is read-only. Execution engine (Phase 4) will process "
        "these approved actions with full logging, checksums, and undo capability."
    )


# ===================================================================
# Router
# ===================================================================
if view == "Drive Overview":
    render_drive_overview()
elif view == "Action Review":
    render_action_review()
elif view == "Execution Plan":
    render_execution_plan()
