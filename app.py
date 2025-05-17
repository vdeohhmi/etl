import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import networkx as nx
from pyvis.network import Network
import plotly.express as px
from openai import OpenAI

# --- App Configuration ---
st.set_page_config(page_title="Data Wizard X", layout="wide")
st.title("🔮 Data Wizard X — Smart ETL and Analysis Studio")

# --- Session State Initialization ---
for key, default in [
    ("datasets", {}), ("current", None),
    ("steps", []), ("versions", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- OpenAI API Key (from ENV) ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Set the OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key)

# --- Helper Functions ---
def load_file(f):
    ext = f.name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            return pd.read_csv(f)
        if ext in ("xls", "xlsx"):
            return pd.read_excel(f, sheet_name=None)
        if ext == "parquet":
            return pd.read_parquet(f)
        if ext == "json":
            return pd.read_json(f)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
    return None

def apply_steps(df):
    st.session_state.versions.append(df.copy())
    for s in st.session_state.steps:
        t = s["type"]
        if t == "rename":
            df = df.rename(columns={s["old"]: s["new"]})
        elif t == "filter" and s.get("expr"):
            try:
                df = df.query(s["expr"])
            except:
                pass
        elif t == "compute" and s.get("expr"):
            try:
                df[s["new"]] = df.eval(s["expr"])
            except:
                df[s["new"]] = df.eval(s["expr"], engine="python")
        elif t == "drop_const":
            df = df.loc[:, df.nunique() > 1]
        elif t == "onehot":
            df = pd.get_dummies(df, columns=s["cols"])
        elif t == "join":
            aux = st.session_state.datasets[s["aux"]]
            df = df.merge(
                aux,
                left_on=s["left"],
                right_on=s["right"],
                how=s["how"]
            )
        elif t == "impute":
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = (
                        df[c].median()
                        if pd.api.types.is_numeric_dtype(df[c])
                        else df[c].mode().iloc[0]
                    )
    return df

# --- UI Tabs ---
tabs = st.tabs([
    "📂 Datasets","✏️ Transform","📈 Profile","💡 Insights",
    "⬇️ Export","🕒 History","⚙️ Snowflake","🤖 AI Toolkit","🕸️ Social Graph"
])

# --- 1. Datasets ---
with tabs[0]:
    st.header("1. Datasets")
    uploads = st.file_uploader(
        "Upload files (CSV/Excel/Parquet/JSON)",
        type=["csv","xls","xlsx","parquet","json"],
        accept_multiple_files=True
    )
    if uploads:
        for f in uploads:
            df = load_file(f)
            if isinstance(df, dict):
                for sheet, sdf in df.items():
                    st.session_state.datasets[f"{f.name}:{sheet}"] = sdf
            elif df is not None:
                st.session_state.datasets[f.name] = df
        st.success("Datasets loaded.")
    if st.session_state.datasets:
        sel = st.selectbox(
            "Select dataset",
            list(st.session_state.datasets.keys())
        )
        st.session_state.current = sel
        st.data_editor(
            st.session_state.datasets[sel],
            key=f"editor_{sel}",
            use_container_width=True
        )

# --- 2. Transform ---
with tabs[1]:
    st.header("2. Transform")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        st.write("**Current Steps:**")
        for i, s in enumerate(st.session_state.steps, 1):
            st.write(f"{i}. {s['type']} — {s.get('desc','')}")
        op = st.selectbox("Operation", [
            "rename","filter","compute","drop_const",
            "onehot","join","impute"
        ])
        with st.form("transform_form", clear_on_submit=False):
            if op == "rename":
                old = st.selectbox("Old column", df.columns)
                new = st.text_input("New column name")
                submitted = st.form_submit_button("Add Rename")
                if submitted:
                    st.session_state.steps.append({
                        "type":"rename","old":old,"new":new,
                        "desc":f"{old}→{new}"
                    })
            elif op == "filter":
                expr = st.text_input("Filter expression")
                submitted = st.form_submit_button("Add Filter")
                if submitted:
                    st.session_state.steps.append({
                        "type":"filter","expr":expr,"desc":expr
                    })
            elif op == "compute":
                newc = st.text_input("New column name")
                desc = st.text_area("Describe logic in plain English")
                submitted = st.form_submit_button("AI Generate & Add Compute")
                if submitted:
                    cols = df.columns.tolist()
                    sample = df.head(3).to_dict("records")
                    prompt = (
                        f"You are a Python data engineer. Columns: {cols}. "
                        f"Sample: {sample}. Generate a pandas eval expression "
                        f"for new column '{newc}' with logic: {desc}. Return only the expression."
                    )
                    with st.spinner("Calling AI…"):
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    expr = resp.choices[0].message.content.strip().strip('"')
                    st.code(expr)
                    st.session_state.steps.append({
                        "type":"compute","new":newc,"expr":expr,"desc":desc
                    })
            elif op == "drop_const":
                submitted = st.form_submit_button("Add Drop Constants")
                if submitted:
                    st.session_state.steps.append({
                        "type":"drop_const","desc":"drop constant cols"
                    })
            elif op == "onehot":
                cols = st.multiselect(
                    "Columns to encode",
                    df.select_dtypes("object").columns
                )
                submitted = st.form_submit_button("Add One-Hot")
                if submitted:
                    st.session_state.steps.append({
                        "type":"onehot","cols":cols,"desc":",".join(cols)
                    })
            elif op == "join":
                aux = st.selectbox(
                    "Aux dataset",
                    [k for k in st.session_state.datasets if k != key]
                )
                left = st.selectbox("Left key", df.columns)
                right = st.selectbox(
                    "Right key",
                    st.session_state.datasets[aux].columns
                )
                how = st.selectbox("Join type", ["inner","left","right","outer"])
                submitted = st.form_submit_button("Add Join")
                if submitted:
                    st.session_state.steps.append({
                        "type":"join","aux":aux,
                        "left":left,"right":right,"how":how,
                        "desc":f"{aux} on {left}={right}"
                    })
            else:  # impute
                submitted = st.form_submit_button("Add Impute")
                if submitted:
                    st.session_state.steps.append({
                        "type":"impute","desc":"auto-impute"
                    })
        # apply and preview
        out = apply_steps(df)
        st.session_state.datasets[key] = out
        st.write("**Transformed Preview:**")
        st.data_editor(
            out,
            key=f"transform_preview_{key}_{len(st.session_state.versions)}",
            use_container_width=True
        )

# --- 3. Profile ---
with tabs[2]:
    st.header("3. Profile")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({
            "dtype": df.dtypes,
            "nulls": df.isna().sum(),
            "null_pct": df.isna().mean() * 100
        })
        st.dataframe(stats, use_container_width=True)

# --- 4. Insights ---
with tabs[3]:
    st.header("4. Insights")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes("number")
        if not num.empty:
            st.plotly_chart(
                px.imshow(num.corr(), text_auto=True),
                use_container_width=True
            )

# --- 5. Export ---
with tabs[4]:
    st.header("5. Export")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox(
            "Format",
            ["CSV","JSON","Parquet","Excel","Snowflake"]
        )
        if st.button("Export Data"):
            if fmt == "CSV":
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode(),
                    "data.csv"
                )
            elif fmt == "JSON":
                st.download_button(
                    "Download JSON",
                    df.to_json(orient="records"),
                    "data.json"
                )
            elif fmt == "Parquet":
                st.download_button(
                    "Download Parquet",
                    df.to_parquet(index=False),
                    "data.parquet"
                )
            elif fmt == "Excel":
                buf = BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "Download Excel",
                    buf.getvalue(),
                    "data.xlsx"
                )
            else:
                tbl = st.text_input("Snowflake table name")
                if st.button("Write to Snowflake"):
                    conn = snowflake.connector.connect(
                        user=st.session_state["sf_user"],
                        password=st.session_state["sf_password"],
                        account=st.session_state["sf_account"],
                        warehouse=st.session_state["sf_warehouse"],
                        database=st.session_state["sf_database"],
                        schema=st.session_state["sf_schema"]
                    )
                    write_pandas(conn, df, tbl)
                    conn.close()
                    st.success(f"Wrote to {tbl}")

# --- 6. History ---
with tabs[5]:
    st.header("6. History")
    key = st.session_state.current
    if key and st.session_state.versions:
        for i, snap in enumerate(st.session_state.versions, 1):
            c1, c2 = st.columns([0.7,0.3])
            c1.write(f"{i}. Saved snapshot")
            if c2.button("Revert", key=f"history_revert_{key}_{i}"):
                st.session_state.datasets[key] = snap
        st.write("**Current Data:**")
        st.data_editor(
            st.session_state.datasets[key],
            key=f"history_preview_{key}_{len(st.session_state.versions)}",
            use_container_width=True
        )

# --- 7. Snowflake Settings ---
with tabs[6]:
    st.header("7. Snowflake Settings")
    st.text_input("Account", key="sf_account")
    st.text_input("User", key="sf_user")
    st.text_input("Password", type="password", key="sf_password")
    st.text_input("Warehouse", key="sf_warehouse")
    st.text_input("Database", key="sf_database")
    st.text_input("Schema", key="sf_schema")

# --- 8. AI Toolkit ---
with tabs[7]:
    st.header("8. AI Toolkit")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset to access AI tools.")
    else:
        df = st.session_state.datasets[key]
        st.subheader("Live Preview")
        st.data_editor(
            df,
            key=f"ai_preview_{key}_{len(st.session_state.steps)}",
            use_container_width=True
        )

        tool = st.selectbox(
            "Choose AI Tool",
            ["Compute Column","Natural Language Query","Data Storytelling"]
        )

        if tool == "Compute Column":
            newc = st.text_input("New column name", key="ai_newc")
            desc = st.text_area("Describe logic in plain English", key="ai_desc")

            with st.form("ai_compute_form", clear_on_submit=False):
                apply_ai = st.form_submit_button("Generate & Apply")
                if apply_ai:
                    cols = df.columns.tolist()
                    sample = df.head(3).to_dict("records")
                    prompt = (
                        f"You are a Python data engineer. Columns: {cols}. "
                        f"Sample: {sample}. Generate a pandas eval expression "
                        f"for new column '{newc}' with logic: {desc}. Return only the expression."
                    )
                    with st.spinner("Calling AI…"):
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    expr = resp.choices[0].message.content.strip().strip('"')
                    st.code(expr)
                    st.session_state.steps.append({
                        "type":"compute","new":newc,"expr":expr,"desc":desc
                    })

            alt = st.text_input(
                "Or paste your own formula:",
                key="ai_alt_formula",
                help="e.g. (df['Ship Date']-df['Order Date']).dt.days"
            )
            if st.button("Apply Alternate Formula"):
                formula = st.session_state.get("ai_alt_formula","").strip()
                if formula:
                    st.session_state.steps.append({
                        "type":"compute","new":newc,"expr":formula,"desc":"manual"
                    })

            df2 = apply_steps(df)
            st.session_state.datasets[key] = df2
            st.subheader("Updated Preview")
            st.data_editor(
                df2,
                key=f"ai_updated_{key}_{len(st.session_state.steps)}",
                use_container_width=True
            )

        elif tool == "Natural Language Query":
            q = st.text_area("Ask a question about your data", key="ai_query")
            if st.button("Run Query"):
                cols = df.columns.tolist()
                sample = df.head(5).to_dict("records")
                prompt = (
                    f"You are a data analyst. Columns: {cols}. Sample: {sample}. "
                    f"Question: {q}. Provide a concise markdown answer with examples or code."
                )
                with st.spinner("AI…"):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}]
                    )
                st.markdown(resp.choices[0].message.content)

        else:  # Data Storytelling
            choice = st.selectbox(
                "Story for:",
                ["Entire Dataset","Single Column"]
            )
            if choice == "Single Column":
                col = st.selectbox("Column", df.columns, key="ai_story_col")
                if st.button("Generate Story"):
                    cols = df.columns.tolist()
                    sample = df.head(5).to_dict("records")
                    prompt = (
                        f"You are a data journalist. Columns: {cols}. Sample: {sample}. "
                        f"Analyze column '{col}': distribution, missing data, outliers."
                    )
                    with st.spinner("AI…"):
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    st.markdown(resp.choices[0].message.content)
            else:
                if st.button("Generate Dataset Story"):
                    cols = df.columns.tolist()
                    sample = df.head(5).to_dict("records")
                    prompt = (
                        f"You are a data journalist. Columns: {cols}. Sample: {sample}. "
                        "Write a detailed report summarizing distributions, correlations, missing data, and use cases."
                    )
                    with st.spinner("AI…"):
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    st.markdown(resp.choices[0].message.content)

# --- 9. Social Graph ---
with tabs[8]:
    st.header("9. Social Network Graph")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        src = st.selectbox("Source column", df.columns, key="sg_src")
        tgt = st.selectbox("Target column", df.columns, key="sg_tgt")
        wt_opt = [None] + list(df.columns)
        wt = st.selectbox("Weight column (optional)", wt_opt, key="sg_wt")
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[src], row[tgt]
                w = float(row[wt]) if wt and pd.notna(row[wt]) else 1.0
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
            edges_sorted = sorted(
                G.edges(data=True),
                key=lambda x: x[2]["weight"],
                reverse=True
            )
            top5 = {(u, v) for u, v, d in edges_sorted[:5]}
            net = Network(
                height="700px", width="100%",
                bgcolor="#222222", font_color="white"
            )
            net.show_buttons(filter_=["physics"])
            for n in G.nodes():
                net.add_node(
                    n,
                    label=str(n),
                    title=f"Degree: {G.degree(n)}",
                    value=G.degree(n)
                )
            for u, v, d in G.edges(data=True):
                net.add_edge(
                    u, v,
                    value=d["weight"],
                    width=4 if (u, v) in top5 or (v, u) in top5 else 1,
                    color="red" if (u, v) in top5 or (v, u) in top5 else "rgba(200,200,200,0.2)",
                    title=f"Weight: {d['weight']}"
                )
            html = net.generate_html()
            import streamlit.components.v1 as components
            components.html(html, height=750, scrolling=True)
