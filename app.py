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

# --- App Config ---
st.set_page_config(page_title="Data Transformer Pro Plus", layout="wide")
st.title("🛠️ Data Transformer Pro Plus — Robust ETL Web App")

# --- Init State ---
for key, default in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helpers ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.get('sf_user',''),
        password=st.session_state.get('sf_password',''),
        account=st.session_state.get('sf_account',''),
        warehouse=st.session_state.get('sf_warehouse',''),
        database=st.session_state.get('sf_database',''),
        schema=st.session_state.get('sf_schema','')
    )

def load_file(u):
    ext = u.name.split('.')[-1].lower()
    if ext == 'csv': return pd.read_csv(u)
    if ext in ['xls','xlsx']: return pd.read_excel(u, sheet_name=None)
    if ext == 'parquet': return pd.read_parquet(u)
    if ext == 'json': return pd.read_json(u)
    return None

def apply_steps(df):
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter' and step['expr']:
            try: df = df.query(step['expr'])
            except: pass
        elif t == 'compute' and step['expr']:
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
                df = df.merge(aux, left_on=step['left'], right_on=step['right'], how=step['how'])
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iloc[0])
    return df

# --- Tabs ---
tabs = st.tabs(["Datasets","Transform","Profile","Insights","Export","History","Snowflake","Pipeline","Social Graph"])

# Datasets
with tabs[0]:
    st.header("1. Datasets")
    files = st.file_uploader("Upload files", accept_multiple_files=True, type=['csv','xls','xlsx','parquet','json'])
    if files:
        for u in files:
            data = load_file(u)
            if isinstance(data, dict):
                for sheet, sdf in data.items():
                    st.session_state.datasets[f"{u.name}:{sheet}"] = sdf
            elif data is not None:
                st.session_state.datasets[u.name] = data
        st.success("Files loaded.")
    if st.session_state.datasets:
        sel = st.selectbox("Select dataset", list(st.session_state.datasets), key='sel')
        st.session_state.current = sel
        st.data_editor(st.session_state.datasets[sel], key=f"editor_{sel}")

# Transform
with tabs[1]:
    st.header("2. Transform")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        for i,s in enumerate(st.session_state.steps): st.write(f"{i+1}. {s['type']} {s.get('desc','')}")
        op = st.selectbox("Op", ['rename','filter','compute','drop_const','onehot','join','impute'], key='op')
        if op=='rename':
            old = st.selectbox("Old", df.columns)
            new = st.text_input("New")
            if st.button("Add"): st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"{old}->{new}"})
        if op=='filter':
            expr = st.text_input("Expr")
            if st.button("Add"): st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        if op=='compute':
            newc = st.text_input("New col")
            expr = st.text_input("Expr/SQL")
            if st.button("Add"): st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':newc})
        if op=='drop_const' and st.button("Add"): st.session_state.steps.append({'type':'drop_const','desc':'drop consts'})
        if op=='onehot':
            cols = st.multiselect("Cols", df.select_dtypes('object').columns)
            if st.button("Add"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
        if op=='join':
            aux = st.selectbox("Aux", [k for k in st.session_state.datasets if k!=key])
            left = st.selectbox("Left", df.columns)
            right = st.selectbox("Right", st.session_state.datasets[aux].columns)
            how = st.selectbox("How", ['inner','left','right','outer'])
            if st.button("Add"): st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':aux})
        if op=='impute' and st.button("Add"): st.session_state.steps.append({'type':'impute','desc':'impute'})
        if st.button("Apply"): st.session_state.datasets[key] = apply_steps(df); st.success("Done")
        st.data_editor(st.session_state.datasets[key], key=f"tr_{key}")

# Profile
with tabs[2]:
    st.header("3. Profile")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        st.dataframe(pd.DataFrame({'dtype':df.dtypes,'nulls':df.isna().sum()}))

# Insights
with tabs[3]:
    st.header("4. Insights")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        num=df.select_dtypes('number')
        if not num.empty: st.plotly_chart(px.imshow(num.corr(),text_auto=True))

# Export
with tabs[4]:
    st.header("5. Export")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        fmt=st.selectbox("Fmt",['CSV','JSON','Parquet','Excel','Snowflake'])
        if st.button("Go"):
            if fmt=='CSV': st.download_button("CSV",df.to_csv(index=False).encode(),"data.csv")
            if fmt=='Excel': out=BytesIO();df.to_excel(out,index=False);st.download_button("XLSX",out.getvalue(),"data.xlsx")
            # Snowflake
            if fmt=='Snowflake':
                tbl=st.text_input("Table");sub=st.button("Write");
                if sub: conn=get_sf_conn();write_pandas(conn,df,tbl);conn.close();st.success("Written")

# History
with tabs[5]:
    st.header("6. History")
    pass

# Snowflake Settings
with tabs[6]:
    st.header("7. Snowflake")
    for field in ['sf_account','sf_user','sf_password','sf_warehouse','sf_database','sf_schema']:
        st.text_input(field, key=field)

# Pipeline YAML
with tabs[7]:
    st.header("8. Pipeline YAML")
    y=yaml.dump({'steps':st.session_state.steps},sort_keys=False)
    st.text_area("YAML",y,height=200);st.download_button("YAML",y,"pipeline.yaml")

# Social Graph
with tabs[8]:
    st.header("9. Social Graph")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        src=st.selectbox("Src",df.columns)
        tgt=st.selectbox("Tgt",df.columns)
        wt=None
        if st.button("Graph"):
            G=nx.Graph()
            for _,r in df.iterrows():G.add_edge(r[src],r[tgt])
            top=sorted(G.edges(),key=lambda e:G.degree(e[0])+G.degree(e[1]),reverse=True)[:5]
            net=Network(height="600px",width="100%",bgcolor="#222222",font_color="white")
            for n in G.nodes():net.add_node(n,label=str(n),value=G.degree(n))
            for u,v in G.edges():
                if (u,v) in top:net.add_edge(u,v,width=3,color='red')
                else:net.add_edge(u,v,width=1,color='rgba(200,200,200,0.2)')
            html=net.generate_html();import streamlit.components.v1 as c; c.html(html,height=650)
