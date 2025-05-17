import os
import re
import uuid
import logging
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
from pandasql import sqldf
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Data Wizard X Pro", layout="wide")
st.title("🔮 Data Wizard X Pro — Next-Level ETL & Analysis")

# --- Session State Initialization ---
def ensure_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
for k, d in [('datasets', {}), ('current', None), ('versions', []), ('ops', [])]:
    ensure_state(k, d)

# --- API Client with Caching ---
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.warning("Set OPENAI_API_KEY in environment to use AI features.")
client = OpenAI(api_key=api_key)

@st.cache_data(show_spinner=False)
def ai_call(prompt: str) -> str:
    logger.info(f"AI prompt: {prompt}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role':'user','content':prompt}]
    )
    return resp.choices[0].message.content.strip()

# --- Helper to clean AI expressions ---
def clean_expr(raw: str) -> str:
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    for l in lines:
        if 'df' in l and '=' not in l:
            return l.strip()
    return next((l for l in reversed(lines) if l.strip()), raw.strip())

# --- File Loading Helper ---
def load_file(f):
    ext = f.name.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'csv': return pd.read_csv(f)
        if ext in ('xls','xlsx'): return pd.read_excel(f, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(f)
        if ext == 'json': return pd.read_json(f)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
    return None

# --- Plugin System for Transform Operations ---
class TransformOp:
    registry = {}
    def __init_subclass__(cls, op_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        TransformOp.registry[op_type] = cls
    def __init__(self, **params):
        self.params = params
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class RenameOp(TransformOp, op_type='rename'):
    def apply(self, df):
        return df.rename(columns={self.params['old']:self.params['new']})

class FilterOp(TransformOp, op_type='filter'):
    def apply(self, df):
        try: return df.query(self.params['expr'])
        except: return df

class ComputeOp(TransformOp, op_type='compute'):
    def apply(self, df):
        new, expr = self.params['new'], self.params['expr']
        try: df[new] = df.eval(expr)
        except: df[new] = df.eval(expr, engine='python')
        return df

class SqlOp(TransformOp, op_type='sql'):
    def apply(self, df):
        sql = self.params['sql']
        engine = create_engine('sqlite:///:memory:')
        df.to_sql('df', engine, index=False)
        try:
            return pd.read_sql(sql, engine)
        except:
            return df

class DropConstOp(TransformOp, op_type='drop_const'):
    def apply(self, df):
        return df.loc[:, df.nunique()>1]

class OneHotOp(TransformOp, op_type='onehot'):
    def apply(self, df):
        return pd.get_dummies(df, columns=self.params.get('cols', []))

class JoinOp(TransformOp, op_type='join'):
    def apply(self, df):
        aux = st.session_state.datasets[self.params['aux']]
        return df.merge(aux,
                        left_on=self.params['left'],
                        right_on=self.params['right'],
                        how=self.params.get('how','inner'))

class ImputeOp(TransformOp, op_type='impute'):
    def apply(self, df):
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(
                    df[c].median() if pd.api.types.is_numeric_dtype(df[c])
                    else df[c].mode().iloc[0]
                )
        return df

# --- Core Pipeline Execution with Parallelism ---
def run_pipeline(df):
    funcs = [TransformOp.registry[o['type']](**o).apply for o in st.session_state.ops]
    result = df.copy()
    with ThreadPoolExecutor() as ex:
        for fn in funcs:
            result = ex.submit(fn, result).result()
    return result

# --- UI Tabs ---
tab_ds, tab_tf, tab_ai, tab_prof, tab_exp, tab_hist, tab_graph = st.tabs([
    '📂 Datasets','✏️ Transforms','🤖 AI Toolkit',
    '📈 Profile','⬇️ Export','🕒 History','🕸️ Social Graph'
])

# 1. Datasets
with tab_ds:
    st.header("1. Datasets")
    files = st.file_uploader("Upload CSV/Excel/Parquet/JSON", accept_multiple_files=True)
    if files:
        for f in files:
            data = load_file(f)
            if isinstance(data, dict):
                for sheet,df_sheet in data.items(): st.session_state.datasets[f"{f.name}:{sheet}"]=df_sheet
            elif isinstance(data,pd.DataFrame): st.session_state.datasets[f.name]=data
        st.success("Datasets loaded.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()))
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"ds_{sel}", use_container_width=True)

# 2. Transforms
with tab_tf:
    st.header("2. Transforms")
    key = st.session_state.current
    if not key:
        st.info("Load a dataset first.")
    else:
        df = st.session_state.datasets[key]
        st.subheader("Before Transformation Preview")
        st.data_editor(df, key=f"preview_before_{key}", use_container_width=True)
        op_types = list(TransformOp.registry.keys())
        op = st.selectbox("Operation", op_types)
        params = {}
        with st.form("form_tf"):
            if op == 'rename':
                params['old'] = st.selectbox("Old column", df.columns)
                params['new'] = st.text_input("New column name")
            elif op == 'filter':
                params['expr'] = st.text_input("Filter expression (pandas query)")
            elif op == 'compute':
                params['new'] = st.text_input("New column name")
                logic = st.text_area("Describe logic (plain English)")
                manual = st.text_input("Or enter expression manually")
                if st.form_submit_button("AI Generate Pandas"):
                    params['expr'] = clean_expr(ai_call(f"Pandas expression for {params['new']}: {logic}"))
                    st.code(params['expr'])
                else:
                    params['expr'] = manual
            elif op == 'sql':
                params['new'] = st.text_input("New column name")
                desc = st.text_area("Describe SQL logic (plain English)")
                manual_sql = st.text_area("Or enter SQL manually")
                if st.form_submit_button("AI Generate SQL"): 
                    params['sql'] = clean_expr(ai_call(f"SQL: SELECT *, {desc} AS {params['new']} FROM df"))
                    st.code(params['sql'])
                else:
                    params['sql'] = manual_sql
            elif op == 'onehot':
                params['cols'] = st.multiselect("Columns to one-hot encode", df.columns)
            elif op == 'join':
                params['aux'] = st.selectbox("Aux dataset", [d for d in st.session_state.datasets if d!=key])
                params['left'] = st.selectbox("Left key", df.columns)
                params['right'] = st.selectbox("Right key", st.session_state.datasets[params['aux']].columns)
                params['how'] = st.selectbox("Join type", ['inner','left','right','outer'])
            submitted = st.form_submit_button("Add Operation")
            if submitted:
                # Validate
                if op in ['filter','compute'] and not params.get('expr'):
                    st.error("Expression is required for compute/filter.")
                elif op == 'sql' and not params.get('sql'):
                    st.error("SQL is required for SQL operations.")
                else:
                    st.session_state.ops.append({'id':str(uuid.uuid4()), 'type':op, **params, 'desc':params.get('expr', params.get('sql',''))})
        st.write("### Pipeline Steps")
        for i, s in enumerate(st.session_state.ops,1): st.write(f"{i}. {s['type']} — {s['desc']}")
        if st.button("Run Pipeline"):
            out = run_pipeline(df)
            st.session_state.datasets[key] = out
            st.success("Pipeline applied.")
            st.data_editor(out, key=f"preview_after_{key}", use_container_width=True)

# 3. AI Toolkit
with tab_ai:
    st.header("3. AI Toolkit")
    st.write("Use the Transforms tab to integrate AI-powered transformations.")

# 4. Profile
with tab_prof:
    st.header("4. Profile")
    df = st.session_state.datasets.get(st.session_state.current)
    if df is not None:
        stats = pd.DataFrame({'dtype':df.dtypes,'nulls':df.isna().sum(),'pct_null':df.isna().mean()*100})
        st.dataframe(stats, use_container_width=True)

# 5. Export
with tab_exp:
    st.header("5. Export")
    df = st.session_state.datasets.get(st.session_state.current)
    if df is not None:
        fmt = st.selectbox("Format", ['CSV','Parquet','Excel','Snowflake'])
        if fmt == 'Snowflake':
            acc = st.text_input('Account'); user=st.text_input('User'); pwd=st.text_input('Password',type='password')
            wh=st.text_input('Warehouse'); db=st.text_input('Database'); schema=st.text_input('Schema'); tbl=st.text_input('Table name')
            if st.button('Write to Snowflake'):
                conn = snowflake.connector.connect(
                    user=user,password=pwd,account=acc,warehouse=wh,database=db,schema=schema
                )
                write_pandas(conn, df, tbl); conn.close(); st.success(f"Written to {tbl}")
        else:
            if st.button("Export Data"):
                buf = BytesIO()
                if fmt=='CSV': st.download_button('CSV', df.to_csv(index=False).encode(), 'data.csv')
                elif fmt=='Parquet': st.download_button('Parquet', df.to_parquet(index=False), 'data.parquet')
                else: df.to_excel(buf,index=False,engine='openpyxl'); st.download_button('Excel',buf.getvalue(),'data.xlsx')

# 6. History
with tab_hist:
    st.header("6. History")
    for i, snap in enumerate(st.session_state.versions,1):
        if st.button(f"Revert to v{i}", key=f"rev_{i}"):
            st.session_state.datasets[st.session_state.current] = snap
            st.experimental_rerun()

# 7. Social Graph
with tab_graph:
    st.header("7. Social Network Graph")
    df = st.session_state.datasets.get(st.session_state.current)
    if df is not None:
        src = st.selectbox('Source column', df.columns, key='sg_src')
        tgt = st.selectbox('Target column', df.columns, key='sg_tgt')
        wt_opt = [None] + list(df.columns)
        wt = st.selectbox('Weight column (optional)', wt_opt, key='sg_wt')
        if st.button('Generate Graph'):
            G = nx.Graph()
            for _, row in df.iterrows():
                u,v = row[src], row[tgt]
                w = float(row[wt]) if wt and pd.notnull(row[wt]) else 1.0
                G.add_edge(u,v, weight=G[u][v]['weight']+w if G.has_edge(u,v) else w)
            edges = sorted(G.edges(data=True), key=lambda x:x[2]['weight'], reverse=True)
            top5 = {(u,v) for u,v,_ in edges[:5]}
            net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')
            net.show_buttons(filter_=['physics'])
            for n in G.nodes(): net.add_node(n, label=str(n), title=f"Degree: {G.degree(n)}", value=G.degree(n))
            for u,v,d in G.edges(data=True):
                style = {'value':d['weight'], 'width':4 if (u,v) in top5 or (v,u) in top5 else 1,
                         'color':'red' if (u,v) in top5 or (v,u) in top5 else 'rgba(200,200,200,0.2)'}
                net.add_edge(u, v, **style)
            import streamlit.components.v1 as components
            components.html(net.generate_html(), height=650, scrolling=True)
