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
from openai import OpenAI

# --- Config ---
st.set_page_config(page_title="Data Wizard X", layout="wide")
st.title("🔮 Data Wizard X — Smart ETL and Analysis Studio")

# --- Session State ---
for k, d in [('datasets', {}), ('current', None), ('steps', []), ('versions', [])]:
    if k not in st.session_state:
        st.session_state[k] = d

# --- OpenAI Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Set OPENAI_API_KEY in your environment before running.")
client = OpenAI(api_key=api_key)

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

def load_file(f):
    ext = f.name.rsplit('.',1)[-1].lower()
    try:
        if ext=='csv': return pd.read_csv(f)
        if ext in ('xls','xlsx'): return pd.read_excel(f, sheet_name=None)
        if ext=='parquet': return pd.read_parquet(f)
        if ext=='json': return pd.read_json(f)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
    return None

def apply_steps(df):
    st.session_state.versions.append(df.copy())
    for step in st.session_state.steps:
        t = step['type']
        if t=='rename':
            df = df.rename(columns={step['old']:step['new']})
        elif t=='filter' and step.get('expr'):
            try: df = df.query(step['expr'])
            except: pass
        elif t=='compute' and step.get('expr'):
            try:
                df[step['new']] = df.eval(step['expr'])
            except:
                try:
                    df[step['new']] = df.eval(step['expr'], engine='python')
                except: pass
        elif t=='drop_const':
            df = df.loc[:, df.nunique()>1]
        elif t=='onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t=='join':
            aux = st.session_state.datasets.get(step['aux'])
            if aux is not None:
                df = df.merge(aux,
                              left_on=step['left'],
                              right_on=step['right'],
                              how=step['how'])
        elif t=='impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c])
                        else df[c].mode().iloc[0]
                    )
    return df

# --- Tabs ---
tabs = st.tabs([
    "📂 Datasets","✏️ Transform","📈 Profile","💡 Insights",
    "⬇️ Export","🕒 History","⚙️ Snowflake","🤖 AI Toolkit","🕸️ Social Graph"
])

# 1. Datasets
with tabs[0]:
    st.header("1. Datasets")
    files = st.file_uploader("Upload (CSV/Excel/Parquet/JSON)", 
                             type=['csv','xls','xlsx','parquet','json'],
                             accept_multiple_files=True)
    if files:
        for f in files:
            df = load_file(f)
            if isinstance(df, dict):
                for name, sdf in df.items():
                    st.session_state.datasets[f"{f.name}:{name}"] = sdf
            elif df is not None:
                st.session_state.datasets[f.name] = df
        st.success("Loaded.")
    if st.session_state.datasets:
        key = st.selectbox("Select dataset", list(st.session_state.datasets.keys()))
        st.session_state.current = key
        st.data_editor(st.session_state.datasets[key], use_container_width=True)

# 2. Transform
with tabs[1]:
    st.header("2. Transform")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        for i, s in enumerate(st.session_state.steps):
            st.write(f"{i+1}. {s['type']} – {s.get('desc','')}")
        op = st.selectbox("Operation", 
            ['rename','filter','compute','drop_const','onehot','join','impute'])
        if op=='rename':
            old = st.selectbox("Old col", df.columns)
            new = st.text_input("New name")
            if st.button("Add Rename"):
                st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"{old}→{new}"})
        elif op=='filter':
            expr=st.text_input("Filter.expr")
            if st.button("Add Filter"):
                st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        elif op=='compute':
            newc=st.text_input("New col name")
            desc=st.text_area("Logic in plain English")
            if st.button("Add Compute"):
                cols=list(df.columns)
                sample=df.head(3).to_dict('records')
                prompt=(f"You are a Python data engineer. Columns:{cols}, sample:{sample}. "
                        f"Generate a pandas eval expression for new column '{newc}' logic:{desc}.")
                with st.spinner("AI…"):
                    resp=client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}]
                    )
                expr=resp.choices[0].message.content.strip().strip('"')
                st.code(expr)
                st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
        elif op=='drop_const':
            if st.button("Add Drop Constants"):
                st.session_state.steps.append({'type':'drop_const','desc':'drop constants'})
        elif op=='onehot':
            cols=st.multiselect("Cols to one-hot", df.select_dtypes('object').columns)
            if st.button("Add One-Hot"):
                st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
        elif op=='join':
            aux=st.selectbox("Aux dataset",[k for k in st.session_state.datasets if k!=key])
            left=st.selectbox("Left key",df.columns)
            right=st.selectbox("Right key",st.session_state.datasets[aux].columns)
            how=st.selectbox("How",['inner','left','right','outer'])
            if st.button("Add Join"):
                st.session_state.steps.append({
                    'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':aux})
        else:
            if st.button("Add Impute"):
                st.session_state.steps.append({'type':'impute','desc':'auto impute'})
        if st.button("Apply Transformations"):
            updated=apply_steps(df)
            st.session_state.datasets[key]=updated
            st.success("Applied.")
        # dynamic key to force refresh
        ver = len(st.session_state.versions)
        st.data_editor(st.session_state.datasets[key],
                       key=f"transformed_{key}_{ver}", use_container_width=True)

# 3. Profile
with tabs[2]:
    st.header("3. Profile")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        stats=pd.DataFrame({
            'dtype':df.dtypes,'nulls':df.isna().sum(),
            'null_pct':df.isna().mean()*100
        })
        st.dataframe(stats,use_container_width=True)

# 4. Insights
with tabs[3]:
    st.header("4. Insights")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        num=df.select_dtypes('number')
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(),text_auto=True),use_container_width=True)

# 5. Export
with tabs[4]:
    st.header("5. Export")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        fmt=st.selectbox("Format",['CSV','JSON','Parquet','Excel','Snowflake'])
        if st.button("Export"):
            if fmt=='CSV':
                st.download_button("CSV",df.to_csv(index=False).encode(),'data.csv')
            elif fmt=='JSON':
                st.download_button("JSON",df.to_json(orient='records'),'data.json')
            elif fmt=='Parquet':
                st.download_button("Parquet",df.to_parquet(index=False),'data.parquet')
            elif fmt=='Excel':
                buf=BytesIO();df.to_excel(buf,index=False,engine='openpyxl')
                st.download_button("Excel",buf.getvalue(),'data.xlsx')
            else:
                tbl=st.text_input("Table name")
                if st.button("Write to Snowflake"):
                    conn=get_sf_conn();write_pandas(conn,df,tbl);conn.close()
                    st.success(f"Wrote to {tbl}")

# 6. History
with tabs[5]:
    st.header("6. History")
    key=st.session_state.current
    if key and st.session_state.versions:
        for idx, snap in enumerate(st.session_state.versions):
            c1,c2=st.columns([0.7,0.3])
            c1.write(f"{idx+1}.")
            if c2.button("Revert",key=f"hist_{idx}"):
                st.session_state.datasets[key]=snap
        # show current
        st.data_editor(st.session_state.datasets[key],
                       key=f"hist_preview_{len(st.session_state.versions)}",
                       use_container_width=True)

# 7. Snowflake Settings
with tabs[6]:
    st.header("7. Snowflake")
    st.text_input("Account",key='sf_account')
    st.text_input("User",key='sf_user')
    st.text_input("Password",type='password',key='sf_password')
    st.text_input("Warehouse",key='sf_warehouse')
    st.text_input("Database",key='sf_database')
    st.text_input("Schema",key='sf_schema')

# 8. AI Toolkit
with tabs[7]:
    st.header("8. AI Toolkit")
    key=st.session_state.current
    if not key:
        st.info("Select a dataset above.")
    else:
        df=st.session_state.datasets[key]
        # live preview with dynamic key
        st.subheader("Data Preview")
        st.data_editor(df, key=f"ai_preview_{len(st.session_state.steps)}", use_container_width=True)

        tool=st.selectbox("Tool",["Compute Column","NL Query","Data Story"])
        if tool=="Compute Column":
            newc=st.text_input("New col name",key='ai_newc')
            desc=st.text_area("Logic (plain English)",key='ai_desc')
            if st.button("Generate & Apply"):
                cols=list(df.columns); sample=df.head(3).to_dict('records')
                prompt=(f"You are a Python data engineer. Columns:{cols}. Sample:{sample}. "
                        f"Generate a pandas eval expression for '{newc}' logic:{desc}.")
                with st.spinner("AI…"):
                    resp=client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}]
                    )
                expr=resp.choices[0].message.content.strip().strip('"')
                # record and apply
                st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
                df2=apply_steps(df); st.session_state.datasets[key]=df2
                st.success(f"Applied: {newc} = `{expr}`")

            # alternate manual box
            st.text_input("Or paste your own formula:",
                          key='alt_formula',
                          help="e.g. (df['Ship Date']-df['Order Date']).dt.days")
            if st.button("Apply Alternate Formula"):
                formula=st.session_state.get('alt_formula','').strip()
                if formula:
                    st.session_state.steps.append({'type':'compute','new':st.session_state['ai_newc'],'expr':formula,'desc':'manual'})
                    df2=apply_steps(df); st.session_state.datasets[key]=df2
                    st.success("Manual formula applied.")

        elif tool=="NL Query":
            q=st.text_area("Question",key='ai_query')
            if st.button("Run Query"):
                cols=list(df.columns); sample=df.head(5).to_dict('records')
                prompt=(f"You are a data analyst. Columns:{cols}. Sample:{sample}. "
                        f"Q:{q}. Give concise markdown answer.")
                with st.spinner("AI…"):
                    resp=client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}]
                    )
                st.markdown(resp.choices[0].message.content)

        else:  # Data Story
            choice=st.selectbox("Story for",["Dataset","Single Column"])
            if choice=="Single Column":
                col=st.selectbox("Column",df.columns)
                if st.button("Generate Story"):
                    cols=list(df.columns); sample=df.head(5).to_dict('records')
                    prompt=(f"Journalist. Columns:{cols}. Sample:{sample}. "
                            f"Analyze '{col}': distribution, missing, outliers.")
                    with st.spinner("AI…"):
                        resp=client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    st.markdown(resp.choices[0].message.content)
            else:
                if st.button("Generate Full Story"):
                    cols=list(df.columns); sample=df.head(5).to_dict('records')
                    prompt=(f"Journalist. Columns:{cols}. Sample:{sample}. "
                            "Write detailed dataset report.")
                    with st.spinner("AI…"):
                        resp=client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}]
                        )
                    st.markdown(resp.choices[0].message.content)

# 9. Social Graph
with tabs[8]:
    st.header("9. Social Graph")
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        src=st.selectbox("Source",df.columns,key='src')
        tgt=st.selectbox("Target",df.columns,key='tgt')
        wt_opt=[None]+list(df.columns)
        wt=st.selectbox("Weight",wt_opt,key='wt')
        if st.button("Build Graph"):
            G=nx.Graph()
            for _,r in df.iterrows():
                u,v=r[src],r[tgt]
                w=float(r[wt]) if wt and pd.notna(r[wt]) else 1.0
                G.add_edge(u,v,weight=G[u][v]['weight']+w if G.has_edge(u,v) else w)
            top5=set([e[:2] for e in sorted(G.edges(data=True),key=lambda x:x[2]['weight'],reverse=True)[:5]])
            net=Network(height="700px",width="100%",bgcolor="#222222",font_color="white")
            net.show_buttons(filter_=['physics'])
            for n in G.nodes():
                net.add_node(n,label=str(n),title=f"Deg:{G.degree(n)}",value=G.degree(n))
            for u,v,d in G.edges(data=True):
                net.add_edge(u,v,value=d['weight'],
                             width=4 if (u,v) in top5 or (v,u) in top5 else 1,
                             color="red" if (u,v) in top5 or (v,u) in top5 else "rgba(200,200,200,0.2)",
                             title=f"W:{d['weight']}")
            html=net.generate_html()
            import streamlit.components.v1 as components
            components.html(html,height=750,scrolling=True)
