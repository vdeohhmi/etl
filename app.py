import os
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from io import BytesIO
from sqlalchemy import create_engine, inspect
from pandasql import sqldf

# --- App Config ---
st.set_page_config(page_title="Data Transformer Pro Plus +", layout="wide")
st.title("🛠️ Data Transformer Pro Plus +")

# Initialize session state
def init_state():
    if 'df' not in st.session_state: st.session_state.df = None
    if 'steps' not in st.session_state: st.session_state.steps = []
    if 'pipeline_loaded' not in st.session_state: st.session_state.pipeline_loaded = False
    if 'df_aux' not in st.session_state: st.session_state.df_aux = None  # for joins
init_state()

# --- File Loading ---
def load_file(u):
    ext = u.name.split('.')[-1].lower()
    try:
        if ext == 'csv': return pd.read_csv(u)
        if ext in ['xls','xlsx']: return pd.read_excel(u, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(u)
        if ext == 'json': return pd.read_json(u)
    except Exception as e:
        st.error(f"Load error: {e}")
    return None

# --- Transformation Functions ---
def rename_columns(df, old, new): return df.rename(columns={old:new})

def filter_rows(df, expr):
    try: return df.query(expr)
    except Exception:
        st.warning(f"Invalid filter: {expr}")
        return df

def compute_column(df, new_col, formula):
    try: df[new_col] = df.eval(formula)
    except Exception:
        st.warning(f"Invalid formula: {formula}")
    return df

def drop_constant(df):
    const_cols = [c for c in df.columns if df[c].nunique()<=1]
    return df.drop(columns=const_cols)

def one_hot_encode(df, cols):
    return pd.get_dummies(df, columns=cols)

def join_data(df, df2, left_on, right_on, how):
    return df.merge(df2, left_on=left_on, right_on=right_on, how=how)

def run_sql(df, query):
    try: return sqldf(query, {'df':df})
    except Exception:
        st.warning(f"Invalid SQL: {query}")
        return df

def auto_impute(df):
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in [np.float64, np.int64]: df[col].fillna(df[col].median(), inplace=True)
            else: df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# --- Step Application ---
def apply_steps(df):
    for i, step in enumerate(st.session_state.steps):
        t = step['type']
        if t=='rename': df = rename_columns(df, step['old'], step['new'])
        elif t=='filter': df = filter_rows(df, step['expr'])
        elif t=='compute': df = compute_column(df, step['new'], step['expr'])
        elif t=='drop_const': df = drop_constant(df)
        elif t=='onehot': df = one_hot_encode(df, step['cols'])
        elif t=='join': df = join_data(df, st.session_state.df_aux, step['left'], step['right'], step['how'])
        elif t=='sql': df = run_sql(df, step['query'])
        elif t=='impute': df = auto_impute(df)
    return df

# --- Pipeline Persistence ---
def save_pipeline(path='pipeline.yaml'):
    with open(path,'w') as f:
        yaml.dump(st.session_state.steps, f)
    st.success(f"Pipeline saved to {path}")

def load_pipeline(path):
    steps = yaml.safe_load(path.read())
    st.session_state.steps = steps
    st.session_state.pipeline_loaded = True
    st.success("Pipeline loaded")

# --- UI Workflow ---
tabs = st.tabs(["📥 Load","🔧 Transform","📊 Profile","⬇️ Export","⚙️ Pipeline"])

# Load tab
with tabs[0]:
    st.subheader("1. Load Data")
    uploaded = st.file_uploader("Upload primary file", type=['csv','xls','xlsx','parquet','json'])
    aux = st.file_uploader("Upload secondary file for join (optional)", type=['csv','xls','xlsx','parquet','json'])
    if uploaded:
        data = load_file(uploaded)
        if isinstance(data, dict): sheet = st.selectbox("Sheet", list(data.keys())); st.session_state.df = data[sheet]
        else: st.session_state.df = data
        st.success("Primary data loaded.")
    if aux:
        aux_data = load_file(aux)
        if isinstance(aux_data, dict): sheet2 = st.selectbox("Aux sheet", list(aux_data.keys())); st.session_state.df_aux = aux_data[sheet2]
        else: st.session_state.df_aux = aux_data
        st.success("Secondary data loaded.")

# Transform tab
with tabs[1]:
    st.subheader("2. Transform Data")
    df = st.session_state.df
    if df is None: st.info("Load data first.")
    else:
        # Manage steps
        for idx, step in enumerate(st.session_state.steps):
            cols = st.columns([0.8,0.1,0.1])
            cols[0].write(f"**{idx+1}. {step['type']}**: {step.get('desc','')}")
            if cols[1].button("⬆", key=f"up{idx}"):
                if idx>0: st.session_state.steps[idx],st.session_state.steps[idx-1] = st.session_state.steps[idx-1],st.session_state.steps[idx]; st.experimental_rerun()
            if cols[2].button("⬇", key=f"down{idx}"):
                if idx<len(st.session_state.steps)-1: st.session_state.steps[idx],st.session_state.steps[idx+1] = st.session_state.steps[idx+1],st.session_state.steps[idx]; st.experimental_rerun()
        st.markdown("---")
        # Add step
        op = st.selectbox("Operation",['rename','filter','compute','drop_const','onehot','join','sql','impute'], key='op_sel')
        if op=='rename':
            old = st.selectbox("Old name", df.columns.tolist()); new = st.text_input("New name")
            if st.button("Add Rename"): st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f"{old}→{new}"})
        if op=='filter':
            expr = st.text_input("Expression (e.g. col>5)")
            if st.button("Add Filter"): st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
        if op=='compute':
            newc = st.text_input("New Col"); expr2 = st.text_input("Formula")
            if st.button("Add Compute"): st.session_state.steps.append({'type':'compute','new':newc,'expr':expr2,'desc':f"{newc}={expr2}"})
        if op=='drop_const':
            if st.button("Add Drop Constant Cols"): st.session_state.steps.append({'type':'drop_const','desc':'drop constant columns'})
        if op=='onehot':
            cols = st.multiselect("Cols to encode", df.select_dtypes('object').columns.tolist())
            if st.button("Add One-Hot"): st.session_state.steps.append({'type':'onehot','cols':cols,'desc':f"onehot {cols}"})
        if op=='join' and st.session_state.df_aux is not None:
            left = st.selectbox("Left key", df.columns.tolist()); right = st.selectbox("Right key", st.session_state.df_aux.columns.tolist())
            how = st.selectbox("Join type",['inner','left','right','outer'])
            if st.button("Add Join"): st.session_state.steps.append({'type':'join','left':left,'right':right,'how':how,'desc':f"join {how} on {left}={right}"})
        if op=='sql':
            query = st.text_area("SQL Query", height=100)
            if st.button("Add SQL"): st.session_state.steps.append({'type':'sql','query':query,'desc':'custom SQL'})
        if op=='impute':
            if st.button("Add Auto Impute"): st.session_state.steps.append({'type':'impute','desc':'auto impute'})
        if st.button("Apply All Steps"): st.session_state.df = apply_steps(df)
        st.data_editor(st.session_state.df, key='transformed')

# Profile tab
with tabs[2]:
    st.subheader("3. Profile Data")
    df = st.session_state.df
    if df is None: st.info("Load and transform data first.")
    else:
        st.dataframe(pd.concat([df.dtypes.rename('dtype'), df.isna().sum().rename('nulls'), (df.isna().mean()*100).rename('null_pct')], axis=1))
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]): st.line_chart(df[col].dropna())

# Export tab
with tabs[3]:
    st.subheader("4. Export Data")
    df = st.session_state.df
    if df is not None:
        fmt = st.selectbox("Format",['CSV','JSON','Parquet','Excel'])
        if st.button("Download"): 
            if fmt=='CSV': data=df.to_csv(index=False).encode(); st.download_button("CSV",data,"data.csv")
            if fmt=='JSON': data=df.to_json(orient='records'); st.download_button("JSON",data,"data.json")
            if fmt=='Parquet': data=df.to_parquet(index=False); st.download_button("Parquet",data,"data.parquet")
            if fmt=='Excel': out=BytesIO(); df.to_excel(out,index=False, engine='openpyxl'); st.download_button("Excel",out.getvalue(),"data.xlsx")
        st.markdown("---")
        dburl = st.text_input("DB URL", os.getenv('DATABASE_URL',''))
        if st.button("Save to DB"): 
            try:
                eng = create_engine(dburl)
                df.to_sql('transformed',eng,if_exists='replace',index=False)
                st.success("Saved to DB table 'transformed'")
            except Exception as e:
                st.error(f"DB error: {e}")

# Pipeline tab
with tabs[4]:
    st.subheader("⚙️ Pipeline Management")
    if st.button("Save Pipeline"): save_pipeline()
    pl_file = st.file_uploader("Load Pipeline YAML", type=['yaml','yml'])
    if pl_file and not st.session_state.pipeline_loaded: load_pipeline(pl_file)
