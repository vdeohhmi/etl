import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pandasql import sqldf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import yaml
import networkx as nx
from pyvis.network import Network
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Session State Initialization ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.get('sf_user',''),
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
        if ext in ['xls','xlsx']: return pd.read_excel(uploader_file, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(uploader_file)
        if ext == 'json': return pd.read_json(uploader_file)
    except Exception as e:
        st.error(f"Failed to load {uploader_file.name}: {e}")
    return None

def apply_steps(df):
    # Snapshot current before transformations
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
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                try: df = sqldf(expr[4:], {'df': df})
                except: pass
            else:
                try: df[step['new']] = df.eval(expr)
                except: pass
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join':
            aux = st.session_state.datasets.get(step['aux'])
            if aux is not None:
                df = df.merge(aux,
                              left_on=step['left'],
                              right_on=step['right'],
                              how=step['how'])
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
    "📜 Pipeline",
    "🕸️ Social Graph"
])

# 1. Datasets
with tabs[0]:
    st.subheader("1. Load Data")
    files = st.file_uploader("Upload files (CSV, Excel, Parquet, JSON)",
                             type=['csv','xls','xlsx','parquet','json'],
                             accept_multiple_files=True)
    if files:
        for u in files:
            data = load_file(u)
            if isinstance(data, dict):
                for sheet, sdf in data.items():
                    st.session_state.datasets[f"{u.name}:{sheet}"] = sdf
            elif data is not None:
                st.session_state.datasets[u.name] = data
        st.success(f"Loaded {len(files)} files.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets.keys()), key='sel')
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}", use_container_width=True)

# 2. Transform
with tabs[1]:
    st.subheader("2. Transform Data")
    key = st.session_state.current
    if not key:
        st.info("Please load a dataset first.")
    else:
        df = st.session_state.datasets[key]
        for i, s in enumerate(st.session_state.steps):
            st.markdown(f"**Step {i+1}:** {s['type']} — {s.get('desc','')} ")
        st.markdown("---")
        op = st.selectbox("Operation", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        if op == 'rename':
            old = st.selectbox("Old column name", df.columns)
            new = st.text_input("New column name")
            if st.button("Add Rename"): st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"Rename {old}→{new}"})
        elif op == 'filter':
            expr = st.text_input("Filter expression (pandas query)")
            if st.button("Add Filter"): st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        elif op == 'compute':
            newc = st.text_input("New column name for compute")
            expr2 = st.text_input("Formula or SQL (prefix SQL:)")
            st.write("**Columns:**", list(df.columns))
            st.write("**Common functions:** np.log(), np.sqrt(), etc.")
            if st.button("Add Compute"): st.session_state.steps.append({'type':'compute','new':newc,'expr':expr2,'desc':f"Compute {newc}"})
        elif op == 'drop_const':
            if st.button("Drop Constant Columns"): st.session_state.steps.append({'type':'drop_const','desc':'Drop constant columns'})
        elif op == 'onehot':
            cols = st.multiselect("Columns to one-hot encode", df.select_dtypes('object').columns)
            if st.button("Add One-Hot"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':f"One-hot {cols}"})
        elif op == 'join':
            aux = st.selectbox("Auxiliary dataset to join", [k for k in st.session_state.datasets if k!=key])
            left = st.selectbox("Left key", df.columns)
            right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
            how = st.selectbox("Join type", ['inner','left','right','outer'])
            if st.button("Add Join"): st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':f"Join {aux}"})
        elif op == 'impute':
            if st.button("Auto Impute Missing"): st.session_state.steps.append({'type':'impute','desc':'Auto-impute missing'})
        if st.button("Apply Transformations"): 
            st.session_state.datasets[key] = apply_steps(df)
            st.success("Transformations applied.")
        st.data_editor(st.session_state.datasets[key], key=f"transformed_{key}", use_container_width=True)

# 3. Profile
with tabs[2]:
    st.subheader("3. Data Profile")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        profile = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'null_pct': df.isna().mean()*100
        })
        st.dataframe(profile, use_container_width=True)

# 4. Insights
with tabs[3]:
    st.subheader("4. Auto Insights")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes('number')
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# 5. Export
with tabs[4]:
    st.subheader("5. Export & Writeback")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox("Export format", ['CSV','JSON','Parquet','Excel','Snowflake'], key='fmt')
        if st.button("Export Now"):
            if fmt=='CSV': st.download_button("Download CSV", df.to_csv(index=False).encode(), "data.csv")
            elif fmt=='JSON': st.download_button("Download JSON", df.to_json(orient='records'), "data.json")
            elif fmt=='Parquet': st.download_button("Download Parquet", df.to_parquet(index=False), "data.parquet")
            elif fmt=='Excel': 
                out=BytesIO(); df.to_excel(out,index=False,engine='openpyxl'); st.download_button("Download Excel", out.getvalue(), "data.xlsx")
            else:
                # auto-create and load Snowflake table
                table_name = st.session_state.current.replace(':','_')
                conn = get_sf_conn()
                cur = conn.cursor()
                # create table DDL
                cols_defs=[]
                for c,dtype in df.dtypes.items():
                    if pd.api.types.is_integer_dtype(dtype): cols_defs.append(f'"{c}" NUMBER')
                    elif pd.api.types.is_float_dtype(dtype): cols_defs.append(f'"{c}" FLOAT')
                    elif pd.api.types.is_datetime64_dtype(dtype): cols_defs.append(f'"{c}" TIMESTAMP_NTZ')
                    else: cols_defs.append(f'"{c}" VARCHAR({int(df[c].astype(str).map(len).max() or 1)})')
                ddl=f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols_defs)})"
                try: cur.execute(ddl)
                except Exception as e: st.error(f"DDL failed: {e}")
                try: write_pandas(conn,df,table_name); st.success(f"Loaded to Snowflake table {table_name}")
                except Exception as e: st.error(f"Write failed: {e}")
                cur.close(); conn.close()

# 6. History
with tabs[5]:
    st.subheader("6. Transformation History")
    if st.session_state.versions:
        for i,(ts,snap) in enumerate(st.session_state.versions):
            cols=st.columns([0.7,0.3])
            cols[0].write(f"{i+1}. {ts}")
            if cols[1].button("Revert", key=f"rev{i}"): st.session_state.datasets[st.session_state.current]=snap; st.experimental_rerun()

# 7. Snowflake Settings
with tabs[6]:
    st.subheader("7. Snowflake Configuration")
    st.text_input("Account", key='sf_account')
    st.text_input("Username", key='sf_user')
    st.text_input("Password", type='password', key='sf_password')
    st.text_input("Warehouse", key='sf_warehouse')
    st.text_input("Database", key='sf_database')
    st.text_input("Schema", key='sf_schema')

# 8. Pipeline YAML
with tabs[7]:
    st.subheader("8. Pipeline as YAML")
    yaml_str=yaml.dump({'pipeline_steps':st.session_state.steps},sort_keys=False)
    st.text_area("YAML",yaml_str,height=300)
    st.download_button("Download YAML",yaml_str,"pipeline.yaml")

# 9. Social Graph
with tabs[8]:
    st.subheader("9. Social Network Graph")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        cols=list(df.columns)
        src=st.selectbox("Source column",cols,key='src')
        tgt=st.selectbox("Target column",cols,key='tgt')
        wt_opt=[None]+cols
        wt=st.selectbox("Weight column",wt_opt,key='wt')
        if st.button("Generate Graph"):
            G=nx.Graph()
            for _,r in df.iterrows():
                u,v=r[src],r[tgt]
                w=float(r[wt]) if wt and pd.notna(r[wt]) else 1
                if G.has_edge(u,v):G[u][v]['weight']+=w
                else:G.add_edge(u,v,weight=w)
            edges=sorted(G.edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
            top5={(u,v)for u,v,_ in edges[:5]}
            net=Network(height='700px',width='100%',bgcolor='#222222',font_color='white')
            net.barnes_hut(gravity=-20000,central_gravity=0.3,spring_length=100,spring_strength=0.001)
            for n in G.nodes():net.add_node(n,label=str(n),title=f'Degree:{G.degree(n)}',value=G.degree(n))
            for u,v,d in G.edges(data=True):
                if (u,v) in top5 or (v,u) in top5:
                    net.add_edge(u,v,value=d['weight'],width=4,color='red',title=f"Weight:{d['weight']}")
                else:
                    net.add_edge(u,v,value=d['weight'],width=1,color='rgba(200,200,200,0.2)',title=f"Weight:{d['weight']}")
            net.show_buttons(filter_=['physics'])
            html=net.generate_html()
            import streamlit.components.v1 as comp; comp.html(html,height=750,scrolling=True)
