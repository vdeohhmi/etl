import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pandasql import sqldf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import openai
import json

# --- App Configuration ---
st.set_page_config(page_title="Data Wizard X", layout="wide")
st.title("🔮 Data Wizard X — Smart ETL and Analysis Studio")

# --- Session State Initialization ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load OpenAI Key ---
if 'openai' in st.secrets:
    openai.api_key = st.secrets['openai'].get('api_key', '')

# --- Helper Functions ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.get('sf_account',''),
        password=st.session_state.get('sf_password',''),
        account=st.session_state.get('sf_account',''),
        warehouse=st.session_state.get('sf_warehouse',''),
        database=st.session_state.get('sf_database',''),
        schema=st.session_state.get('sf_schema','')
    )

def load_file(uploader_file):
    ext = uploader_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv': return pd.read_csv(uploader_file)
        if ext in ['xls', 'xlsx']: return pd.read_excel(uploader_file, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(uploader_file)
        if ext == 'json': return pd.read_json(uploader_file)
    except Exception as e:
        st.error(f"Failed to load {uploader_file.name}: {e}")
    return None

def apply_steps(df):
    ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.versions.append((ts, df.copy()))
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter' and step.get('expr'):
            try: df = df.query(step['expr'])
            except: pass
        elif t == 'compute' and step.get('expr'):
            try: df[step['new']] = df.eval(step['expr'])
            except:
                try: df[step['new']] = df.eval(step['expr'], engine='python')
                except: pass
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join':
            aux = st.session_state.datasets.get(step['aux'])
            if aux is not None:
                df = df.merge(aux, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0]
                    )
    return df

# --- UI Tabs ---
tabs = st.tabs([
    "📂 Datasets",
    "✏️ Transform",
    "📈 Profile",
    "💡 Insights",
    "⬇️ Export",
    "🕒 History",
    "⚙️ Snowflake",
    "🤖 AI Toolkit",
    "🕸️ Social Graph"
])

# --- 1. Datasets ---
with tabs[0]:
    st.header("1. Datasets")
    uploads = st.file_uploader("Upload files (CSV/Excel/Parquet/JSON)", type=['csv','xls','xlsx','parquet','json'], accept_multiple_files=True)
    if uploads:
        for u in uploads:
            data = load_file(u)
            if isinstance(data, dict):
                for sheet, sdf in data.items():
                    st.session_state.datasets[f"{u.name}:{sheet}"] = sdf
            elif data is not None:
                st.session_state.datasets[u.name] = data
        st.success("Datasets loaded.")
    if st.session_state.datasets:
        key = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key='sel_dataset')
        st.session_state.current = key
        st.data_editor(st.session_state.datasets[key], key=f"editor_{key}", use_container_width=True)

# --- 2. Transform ---
with tabs[1]:
    st.header("2. Transform")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        for i, step in enumerate(st.session_state.steps): st.write(f"{i+1}. {step['type']} — {step.get('desc','')}")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        if op == 'rename':
            old = st.selectbox("Old column", df.columns); new = st.text_input("New column name")
            if st.button("Add Rename"): st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"Rename {old}→{new}"})
        elif op == 'filter':
            expr = st.text_input("Filter expression")
            if st.button("Add Filter"): st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        elif op == 'compute':
            newc = st.text_input("New column name"); desc = st.text_area("Describe logic in plain English")
            if st.button("AI Generate & Add Compute"):
                if not openai.api_key:
                    st.error("OpenAI key not set in .streamlit/secrets.toml")
                else:
                    cols = df.columns.tolist(); sample = df.head(3).to_dict(orient='records')
                    prompt = (
                        f"You are a Python data engineer. DataFrame columns: {cols}. Sample rows: {sample}. "
                        f"Generate a pandas eval expression for new column '{newc}' with logic: {desc}. "
                        "Return only the expression string."
                    )
                    with st.spinner("Generating expression..."):
                        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
                    expr = resp.choices[0].message.content.strip().strip('"')
                    st.code(f"df['{newc}'] = df.eval('{expr}')")
                    st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
        elif op == 'drop_const':
            if st.button("Add Drop Constants"): st.session_state.steps.append({'type':'drop_const','desc':'Drop constants'})
        elif op == 'onehot':
            cols = st.multiselect("Columns to encode", df.select_dtypes('object').columns)
            if st.button("Add One-Hot"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
        elif op == 'join':
            aux = st.selectbox("Aux dataset", [k for k in st.session_state.datasets if k != key])
            left = st.selectbox("Left key", df.columns); right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
            how = st.selectbox("Join type", ['inner','left','right','outer'])
            if st.button("Add Join"): st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':aux})
        elif op == 'impute':
            if st.button("Add Impute"): st.session_state.steps.append({'type':'impute','desc':'Auto-impute'})
        if st.button("Apply Transformations"): st.session_state.datasets[key] = apply_steps(df); st.success("Transformations applied.")
        st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# --- 3. Profile ---
with tabs[2]:
    st.header("3. Profile")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({'dtype': df.dtypes, 'nulls': df.isna().sum(), 'null_pct': df.isna().mean()*100})
        st.dataframe(stats, use_container_width=True)

# --- 4. Insights ---
with tabs[3]:
    st.header("4. Insights")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes('number')
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# --- 5. Export ---
with tabs[4]:
    st.header("5. Export")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox("Format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt_export')
        if st.button("Export"):
            if fmt == 'CSV': st.download_button("Download CSV", df.to_csv(index=False).encode(), "data.csv")
            elif fmt == 'JSON': st.download_button("Download JSON", df.to_json(orient='records'), "data.json")
            elif fmt == 'Parquet': st.download_button("Download Parquet", df.to_parquet(index=False), "data.parquet")
            elif fmt == 'Excel':
                out = BytesIO(); df.to_excel(out, index=False, engine='openpyxl'); st.download_button("Download Excel", out.getvalue(), "data.xlsx")
            else:
                tbl = st.text_input("Snowflake table name", key='exp_tbl')
                if st.button("Write to Snowflake"): conn = get_sf_conn(); write_pandas(conn, df, tbl); conn.close(); st.success(f"Written to {tbl}")

# --- 6. History ---
with tabs[5]:
    st.header("6. History")
    if st.session_state.versions:
        for idx,(ts,snap) in enumerate(st.session_state.versions): cols = st.columns([0.7,0.3]); cols[0].write(f"{idx+1}. {ts}");
            if cols[1].button("Revert", key=f"hist_{idx}"): st.session_state.datasets[st.session_state.current]=snap; st.experimental_rerun()

# --- 7. Snowflake Settings ---
with tabs[6]:
    st.header("7. Snowflake Settings")
    st.text_input("Account", key='sf_account')
    st.text_input("User", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

# --- 8. AI Toolkit ---
with tabs[7]:
    st.header("8. AI Toolkit")
    key = st.session_state.current
    if not key:
        st.info("Select a dataset to access AI tools.")
    elif not openai.api_key:
        st.warning("Configure your OpenAI key in .streamlit/secrets.toml to use AI features.")
    else:
        df = st.session_state.datasets[key]
        tool = st.selectbox("Choose AI Tool:", ["Compute Column", "Natural Language Query", "Data Storytelling"])
        # Compute Column tool
        if tool == "Compute Column":
            newc = st.text_input("New column name")
            desc = st.text_area("Describe logic in plain English")
            if st.button("Generate Formula"):
                cols = df.columns.tolist(); sample = df.head(3).to_dict(orient='records')
                prompt = (
                    f"You are a Python data engineer. Columns: {cols}. Sample: {sample}. "
                    f"Generate a pandas eval expression for new column '{newc}' with logic: {desc}. Return only the expression."
                )
                with st.spinner("Generating..."):
                    resp = openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
                expr = resp.choices[0].message.content.strip().strip('"')
                st.code(f"df['{newc}'] = df.eval('{expr}')")
                st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
        # Natural Language Query tool
        elif tool == "Natural Language Query":
            query = st.text_area("Ask a question about your data")
            if st.button("Run Query"):
                cols = df.columns.tolist(); sample = df.head(5).to_dict(orient='records')
                prompt = (
                    f"You are a data analyst. Columns: {cols}. Sample: {sample}. "
                    f"Question: {query}. Provide a concise markdown answer with examples or code."  )
                with st.spinner("Querying..."):
                    resp = openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
                st.markdown(resp.choices[0].message.content)
        # Data Storytelling tool
        else:
            summary_type = st.selectbox("Story for:", ["Entire Dataset", "Single Column"])
            if summary_type == "Single Column":
                col = st.selectbox("Column", df.columns)
                if st.button("Generate Story"):
                    cols = df.columns.tolist(); sample = df.head(5).to_dict(orient='records')
                    prompt = (
                        f"You are a data journalist. Columns: {cols}. Sample: {sample}. "
                        f"Analyze column '{col}': discuss distribution, missing data, outliers, implications. Provide markdown."  )
                    with st.spinner("Writing story..."):
                        resp = openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
                    st.markdown(resp.choices[0].message.content)
            else:
                if st.button("Generate Dataset Story"):
                    cols = df.columns.tolist(); sample = df.head(5).to_dict(orient='records')
                    prompt = (
                        f"You are a data journalist. Columns: {cols}. Sample: {sample}. "
                        "Write a detailed report summarizing key insights: distributions, correlations, missing data, business use cases. Markdown." )
                    with st.spinner("Writing report..."):
                        resp = openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
                    st.markdown(resp.choices[0].message.content)

# --- 9. Social Graph ---
with tabs[8]:
    st.header("9. Social Network Graph")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        src = st.selectbox("Source column", df.columns, key='src_col')
        tgt = st.selectbox("Target column", df.columns, key='tgt_col')
        wt_opt = [None] + list(df.columns)
        wt = st.selectbox("Weight column (optional)", wt_opt, key='wt_col')
        if st.button("Generate Graph"):
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[src], row[tgt]
                w = float(row[wt]) if wt and pd.notna(row[wt]) else 1.0
                if G.has_edge(u, v): G[u][v]['weight'] += w
                else: G.add_edge(u, v, weight=w)
            edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            top5 = {(u, v) for u, v, d in edges_sorted[:5]}
            net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
            net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=100, spring_strength=0.001)
            for n in G.nodes(): net.add_node(n, label=str(n), title=f"Degree: {G.degree(n)}", value=G.degree(n))
            for u, v, data in G.edges(data=True):
                if (u, v) in top5 or (v, u) in top5:
                    net.add_edge(u, v, value=data['weight'], width=4, color='red', title=f"Weight: {data['weight']}")
                else:
                    net.add_edge(u, v, value=data['weight'], width=1, color='rgba(200,200,200,0.2)', title=f"Weight: {data['weight']}")
            net.show_buttons(filter_=['physics'])
            html = net.generate_html()
            import streamlit.components.v1 as components
            components.html(html, height=750, scrolling=True)
