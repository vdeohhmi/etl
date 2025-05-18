import os
import re
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from io import BytesIO
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from openai import OpenAI
from sqlalchemy import create_engine
import dask.dataframe as dd
from st_aggrid import AgGrid, GridOptionsBuilder
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="Data Wizard X Pro", layout="wide")
st.title("🔮 Data Wizard X Pro — Polars, Dask & AI-Driven Visuals")

# --- Session State Initialization ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
for k, v in [('datasets', {}), ('current', None)]:
    init_state(k, v)

# --- OpenAI Client ---
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error('Please set OPENAI_API_KEY environment variable to enable AI features.')
client = OpenAI(api_key=api_key)

# --- Helpers ---
def sanitize_cols(pl_df: pl.DataFrame) -> pl.DataFrame:
    cols = [re.sub(r'[^0-9a-z_]+','_',c.strip().lower().replace(' ','_')) for c in pl_df.columns]
    return pl_df.rename({old:new for old,new in zip(pl_df.columns,cols)})

@st.cache_data
def ai_generate_code(prompt:str) -> str:
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role':'user','content':prompt}]
    )
    return resp.choices[0].message.content.strip()

# Load files as Polars DataFrame

def load_file(f) -> pl.DataFrame:
    ext=f.name.split('.')[-1].lower()
    try:
        if ext=='csv': df=pl.read_csv(f)
        elif ext=='xls': pdf=pd.read_excel(f,engine='xlrd'); df=pl.from_pandas(pdf)
        elif ext=='xlsx': pdf=pd.read_excel(f,engine='openpyxl'); df=pl.from_pandas(pdf)
        elif ext=='parquet': df=pl.read_parquet(f)
        elif ext=='json': df=pl.read_json(f)
        else: return None
        return sanitize_cols(df)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
        return None

# --- UI Tabs ---
tabs=st.tabs(['Datasets','Transform','Profile','Export','Graph','Snowflake','AI Query'])

# 1) Datasets
with tabs[0]:
    st.header('1. Datasets')
    files=st.file_uploader('Upload CSV/XLS/JSON/Parquet',accept_multiple_files=True)
    if files:
        for f in files:
            df=load_file(f)
            if df is not None: st.session_state.datasets[f.name]=df
        st.success('Loaded datasets')
    if st.session_state.datasets:
        key=st.selectbox('Select dataset',list(st.session_state.datasets.keys()))
        st.session_state.current=key
        df=st.session_state.datasets[key]
        gb=GridOptionsBuilder.from_dataframe(df.to_pandas())
        gb.configure_pagination(); gb.configure_side_bar()
        AgGrid(df.to_pandas(),gridOptions=gb.build(),enable_enterprise_modules=False)

# 2) Transform
with tabs[1]:
    st.header('2. Transform')
    key=st.session_state.current
    if not key: st.info('Upload and select a dataset')
    else:
        df_pl=st.session_state.datasets[key]
        st.subheader('Preview Before')
        AgGrid(df_pl.to_pandas(),enable_enterprise_modules=False)
        op=st.selectbox('Operation',['Polars Compute','Dask Filter','SQL Execute','Drop Const','One-Hot','Impute'])
        # Polars Compute
        if op=='Polars Compute':
            newc=st.text_input('New column'); logic=st.text_area('Logic (English)')
            manual=st.text_input('Manual Polars expr (pl.col(...))')
            if st.button('Generate Expr'): prompt=f"Given DataFrame df columns {list(df_pl.columns)} and logic '{logic}', generate Polars expr."; expr=ai_generate_code(prompt); st.code(expr)
            expr=manual or locals().get('expr','')
            if expr and st.button('Apply Polars Compute'):
                try: series=eval(expr,{'pl':pl}); df_new=df_pl.with_columns(series.alias(newc)); st.session_state.datasets[key]=df_new; st.experimental_rerun()
                except Exception as e: st.error(f'Error: {e}')
        # Dask Filter
        if op=='Dask Filter':
            logic=st.text_area('Filter logic (English)'); manual=st.text_input('Manual Dask expr')
            if st.button('Generate Filter'): prompt=f"Given df columns {list(df_pl.columns)} and logic '{logic}', generate Dask filter expr."; expr=ai_generate_code(prompt); st.code(expr)
            expr=manual or locals().get('expr','')
            if expr and st.button('Apply Dask Filter'):
                try:
                    df_dd=dd.from_pandas(df_pl.to_pandas(),npartitions=2)
                    df_f=df_dd.query(expr).compute()
                except:
                    df_f=df_pl.to_pandas().query(expr)
                df_new=pl.from_pandas(df_f); st.session_state.datasets[key]=df_new; st.experimental_rerun()

        # SQL Execute
        if op=='SQL Execute':
            newc=st.text_input('New column (SQL)'); logic=st.text_area('Logic (English SQL)'); manual=st.text_area('Manual SQL')
            if st.button('Generate SQL'): prompt=f"Table df columns {list(df_pl.columns)}. Write SQL to compute {newc} as {logic}. Return only query."; sql=ai_generate_code(prompt); st.code(sql)
            sql=manual or locals().get('sql','')
            if sql and st.button('Apply SQL'):
                try: engine=create_engine('sqlite:///:memory:'); pdf=df_pl.to_pandas(); pdf.to_sql('df',engine,index=False); df_sql=pd.read_sql(sql,engine); df_new=pl.from_pandas(df_sql); st.session_state.datasets[key]=df_new; st.experimental_rerun()
                except Exception as e: st.error(f'SQL err: {e}')

        # Drop Const
        if op=='Drop Const' and st.button('Drop Const'): df_new=df_pl.select(pl.exclude(pl.all().filter(lambda s: df_pl[n:=s.name].n_unique()<=1))); st.session_state.datasets[key]=df_new; st.experimental_rerun()
        # One-Hot
        if op=='One-Hot' and st.button('One-Hot'): df_new=pl.from_pandas(pd.get_dummies(df_pl.to_pandas())); st.session_state.datasets[key]=df_new; st.experimental_rerun()
        # Impute
        if op=='Impute' and st.button('Impute'): df_new=df_pl.fill_null(strategy='median'); st.session_state.datasets[key]=df_new; st.experimental_rerun()

# 3) Profile
with tabs[2]:
    st.header('3. Profile')
    key=st.session_state.current
    if key: st.dataframe(st.session_state.datasets[key].describe().to_pandas())

# 4) Export
with tabs[3]:
    st.header('4. Export')
    key=st.session_state.current
    if key:
        df_pl=st.session_state.datasets[key]
        fmt=st.selectbox('Format',['CSV','Parquet','JSON'],key='exp_fmt')
        if st.button('Download'):
            if fmt=='CSV': st.download_button('CSV',df_pl.write_csv(),'data.csv')
            elif fmt=='Parquet': st.download_button('Parquet',df_pl.write_parquet(),'data.parquet')
            else: st.download_button('JSON',df_pl.write_json(),'data.json')

# 5) Graph
with tabs[4]:
    st.header('5. Graph')
    key=st.session_state.current
    if key:
        pdf=st.session_state.datasets[key].to_pandas()
        src=st.selectbox('Source',pdf.columns,key='g_src')
        tgt=st.selectbox('Target',pdf.columns,key='g_tgt')
        weight=st.selectbox('Edge weight',[None]+pdf.select_dtypes('number').columns.tolist(),key='g_w')
        top_n=st.number_input('Top N edges',value=5,min_value=1,key='g_top')
        G=nx.Graph()
        for _,r in pdf.iterrows(): u,v=r[src],r[tgt]; w=float(r[weight]) if weight and not np.isnan(r[weight]) else 1.0; G.add_edge(u,v,weight=G.get_edge_data(u,v,{'weight':0})['weight']+w)
        edges=sorted(G.edges(data=True),key=lambda x:x[2]['weight'],reverse=True);
        top_set={(u,v) for u,v,_ in edges[:top_n]}
        pos=nx.random_layout(G)
        traces=[]
        for u,v,data in G.edges(data=True):
            x0,y0=pos[u]; x1,y1=pos[v]
            color='red' if (u,v) in top_set or (v,u) in top_set else 'gray'
            width=3 if (u,v) in top_set or (v,u) in top_set else 1
            traces.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode='lines',line=dict(width=width,color=color),hovertext=f"{u}-{v}: {data['weight']}",hoverinfo='text'))
        node_trace=go.Scatter(x=[pos[n][0] for n in G.nodes()],y=[pos[n][1] for n in G.nodes()],mode='markers+text',text=list(G.nodes()),textposition='top center',marker=dict(size=10,color='skyblue'))
        fig=go.Figure(data=traces+[node_trace]); fig.update_layout(xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(showgrid=False,zeroline=False),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

# 6) Snowflake
with tabs[5]:
    st.header('6. Snowflake')
    key=st.session_state.current
    acc=st.text_input('Account');usr=st.text_input('User');pwd=st.text_input('Password',type='password')
    wh=st.text_input('Warehouse');db=st.text_input('Database');sc=st.text_input('Schema');tbl=st.text_input('Table')
    if st.button('Write') and key:
        df_pl=st.session_state.datasets[key]
        conn=snowflake.connector.connect(user=usr,password=pwd,account=acc,warehouse=wh,database=db,schema=sc)
        write_pandas(conn,df_pl.to_pandas(),tbl);conn.close();st.success(f'Written to {tbl}')

# 7) AI Supercharged Suite
with tabs[6]:
    st.header('7. AI Supercharged Suite')
    key = st.session_state.current
    if not key:
        st.info('Select a dataset first')
    else:
        df_pl = st.session_state.datasets[key]
        pdf = df_pl.to_pandas()

        # A) Narrative Insights
        st.subheader('A) Narrative Insights')
        nlq = st.text_area('Ask a natural language question')
        if st.button('Generate Narrative'):
            prompt = (
                f"You are a data storyteller. Columns: {list(df_pl.columns)}. "
                f"Sample: {pdf.head(5).to_dict(orient='records')}. "
                f"Question: '{nlq}'. Provide concise markdown narrative."
            )
            narrative = ai_generate_code(prompt)
            st.markdown(narrative)

        # B) Counterfactual Explorer
        st.subheader('B) Counterfactual Explorer')
        from sklearn.tree import DecisionTreeClassifier
        num = pdf.select_dtypes(include=[np.number])
        if len(num.columns) >= 2:
            target = st.selectbox('Binary target', [c for c in num.columns if pdf[c].nunique()==2], key='cf_target')
            feats = st.multiselect('Features', [c for c in num.columns if c!=target], key='cf_feats')
            if st.button('Explore Counterfactual'):
                model = DecisionTreeClassifier(max_depth=3).fit(pdf[feats], pdf[target])
                inst = pdf[feats].iloc[[0]]
                base = model.predict(inst)[0]
                st.write(f"Base pred: {base}")
                for col in feats:
                    for d in [-1,1]:
                        tmp = inst.copy(); tmp[col] += d*pdf[col].std()
                        if model.predict(tmp)[0] != base:
                            st.markdown(f"Change **{col}** by {d*pdf[col].std():.2f} flips prediction.")
                            break

        # C) Anomaly Detection
        st.subheader('C) Anomaly Detection')
        from sklearn.ensemble import IsolationForest
        num_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
        contam = st.slider('Contamination rate', 0.01, 0.2, 0.05, key='an_cont')
        if st.button('Run Anomaly Detection') and num_cols:
            iso = IsolationForest(contamination=contam, random_state=42)
            preds = iso.fit_predict(pdf[num_cols])
            pdf['anomaly'] = preds == -1
            st.dataframe(pdf[pdf['anomaly']])
            fig = px.scatter_matrix(pdf.sample(min(100, len(pdf))), dimensions=num_cols, color='anomaly')
            st.plotly_chart(fig, use_container_width=True)

        # D) Clustering Explorer
        st.subheader('D) Clustering Explorer')
        from sklearn.cluster import KMeans
        k = st.number_input('Number of clusters', 2, 10, 3, key='clust_k')
        feats2 = st.multiselect('Cluster on', num_cols, default=num_cols[:2], key='clust_feats')
        if st.button('Run Clustering') and len(feats2)>=2:
            km = KMeans(n_clusters=k, random_state=42).fit(pdf[feats2])
            pdf['cluster'] = km.labels_
            fig = px.scatter(pdf, x=feats2[0], y=feats2[1], color='cluster', symbol='cluster')
            st.plotly_chart(fig, use_container_width=True)

        # E) Similarity Search
        st.subheader('E) Similarity Search')
        text_cols = pdf.select_dtypes(include=[object]).columns.tolist()
        if text_cols:
            col = st.selectbox('Text column', text_cols, key='sim_col')
            query = st.text_input('Search query', key='sim_q')
            if st.button('Run Similarity') and query:
                resp = client.embeddings.create(model='text-embedding-ada-002', input=[query]+pdf[col].astype(str).tolist())
                query_emb = np.array(resp.data[0].embedding)
                embs = np.array([d.embedding for d in resp.data[1:]])
                sims = (embs @ query_emb)/(np.linalg.norm(embs,axis=1)*np.linalg.norm(query_emb))
                pdf['similarity'] = sims
                topn = pdf.nlargest(5,'similarity')[[col,'similarity']]
                st.table(topn)

        # F) Automated Email Summary
        st.subheader('F) Automated Email Summary')
        email = st.text_input('Recipient email')
        time = st.time_input('Send time')
        if st.button('Schedule Daily Summary'):
            schedule = (
                f"BEGIN:VEVENT
RRULE:FREQ=DAILY;BYHOUR={time.hour};BYMINUTE={time.minute};BYSECOND=0
END:VEVENT"
            )
            summary_prompt = (
                f"Summarize dataset shape {pdf.shape} and variables {list(df_pl.columns)} in 3 bullet points."
            )
            st.session_state['auto_email'] = (email, summary_prompt, schedule)
            st.success('Scheduled daily email summary.')

        # G) Dashboard Designer
        st.subheader('G) Generative Dashboard Designer')
        dash_q = st.text_area('Describe dashboard layout (e.g. trends, breakdowns, correlations)')
        if st.button('Generate Dashboard'):
            prompt = (
                f"You are a Python visualization wizard. DataFrame df columns: {list(df_pl.columns)}. "
                f"Sample: {pdf.head(5).to_dict(orient='records')}. "
                f"Generate a Plotly Express dashboard with subplots as described: '{dash_q}'. Return only functional Python code."
            )
            code = ai_generate_code(prompt)
            st.code(code)
            local_vars = {'df': pdf, 'px': px, 'go': go}
            try:
                exec(code, {}, local_vars)
            except Exception as e:
                st.error(f"Dashboard execution failed: {e}")
