import os
import json
import streamlit as st
import pandas as pd
from io import BytesIO
from sqlalchemy import create_engine, inspect

# --- Configuration ---
st.set_page_config(page_title="Advanced Business Data Transformer", layout="wide")
if 'steps' not in st.session_state:
    st.session_state.steps = []
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None

# --- File Loading ---
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return {'Sheet1': pd.read_csv(uploaded_file)}
    if name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file, sheet_name=None)
    if name.endswith('.json'):
        return {'Data': pd.read_json(uploaded_file)}
    if name.endswith('.parquet'):
        return {'Data': pd.read_parquet(uploaded_file)}
    st.error('Unsupported file type')
    return {}

# --- Cleaning Utilities ---
def auto_clean(df):
    # Trim whitespace & lowercase text
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
    # Fill numeric nulls with median
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def outlier_removal(df, columns, threshold):
    for col in columns:
        if col in df:
            mean, std = df[col].mean(), df[col].std()
            df = df[(df[col] - mean).abs() <= threshold * std]
    return df

# --- Transformation Application ---
def apply_steps(df):
    for step in st.session_state.steps:
        op = step['operation']
        if op == 'rename':
            df = df.rename(columns=step['mapping'])
        elif op == 'filter':
            df = df.query(step['expression'])
        elif op == 'compute':
            df[step['new_column']] = df.eval(step['expression'])
        elif op == 'drop':
            df = df.drop(columns=step['columns'])
        elif op == 'sort':
            df = df.sort_values(by=step['columns'], ascending=step['ascending'])
        elif op == 'pivot':
            df = df.pivot_table(index=step['index'], columns=step['columns'], values=step['values'], aggfunc=step['aggfunc'])
        elif op == 'melt':
            df = pd.melt(df, id_vars=step['id_vars'], value_vars=step['value_vars'])
        elif op == 'trim':
            for col in step['columns']:
                df[col] = df[col].str.strip()
        elif op == 'case':
            func = step['case']; cols = step['columns']
            for col in cols:
                if func == 'lower': df[col] = df[col].str.lower()
                if func == 'upper': df[col] = df[col].str.upper()
                if func == 'title': df[col] = df[col].str.title()
        elif op == 'dedupe':
            df = df.drop_duplicates(subset=step['subset'] or None, keep=step['keep'])
        elif op == 'impute':
            for col in step['columns']:
                strat = step['strategy']
                if strat == 'mean': df[col].fillna(df[col].mean(), inplace=True)
                elif strat == 'median': df[col].fillna(df[col].median(), inplace=True)
                elif strat == 'mode': df[col].fillna(df[col].mode()[0], inplace=True)
                elif strat == 'constant': df[col].fillna(step.get('constant'), inplace=True)
        elif op == 'parse_date':
            for col in step['columns']:
                df[col] = pd.to_datetime(df[col], format=step['format'], errors='coerce')
        elif op == 'remove_special':
            for col in step['columns']:
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        elif op == 'outlier':
            df = outlier_removal(df, step['columns'], step['threshold'])
        elif op == 'auto_clean':
            df = auto_clean(df)
    return df

# --- UI ---
st.title('🚀 Advanced Data Transformer')

uploaded = st.file_uploader('Upload file', type=['csv','xls','xlsx','json','parquet'])
if not uploaded:
    st.info('Please upload a file to begin')
    st.stop()

datasets = load_file(uploaded)
sheet = st.selectbox('Select sheet/dataset', list(datasets.keys()))
df_orig = datasets[sheet].copy()

# Sidebar: Steps
st.sidebar.header('Transformation Steps')
for i, s in enumerate(st.session_state.steps):
    st.sidebar.write(f"{i+1}. {s['operation']} - {s.get('description','')}  ")
with st.sidebar.expander('Add Step'):
    op = st.selectbox('Operation', [
        'rename','filter','compute','drop','sort','pivot','melt',
        'trim','case','dedupe','impute','parse_date','remove_special','outlier','auto_clean'
    ])
    if op == 'rename':
        mapping = st.text_area('JSON mapping', value='{}')
        if st.button('Add'): st.session_state.steps.append({'operation':'rename','mapping':json.loads(mapping),'description':mapping})
    elif op == 'filter':
        expr = st.text_input('Expression')
        if st.button('Add'): st.session_state.steps.append({'operation':'filter','expression':expr,'description':expr})
    elif op == 'compute':
        new = st.text_input('Column')
        expr = st.text_area('Expression')
        if st.button('Add'): st.session_state.steps.append({'operation':'compute','new_column':new,'expression':expr,'description':f"{new}={expr}"})
    elif op == 'drop':
        cols = st.multiselect('Columns', df_orig.columns.tolist())
        if st.button('Add'): st.session_state.steps.append({'operation':'drop','columns':cols,'description':f"drop {cols}"})
    elif op == 'sort':
        cols = st.multiselect('By', df_orig.columns.tolist())
        asc = st.checkbox('Ascending', value=True)
        if st.button('Add'): st.session_state.steps.append({'operation':'sort','columns':cols,'ascending':asc,'description':f"sort {cols} asc={asc}"})
    elif op == 'pivot':
        idx = st.multiselect('Index', df_orig.columns.tolist())
        cols = st.multiselect('Columns', df_orig.columns.tolist())
        vals = st.selectbox('Values', df_orig.columns.tolist())
        agg = st.selectbox('Agg', ['sum','mean','min','max','count'])
        if st.button('Add'): st.session_state.steps.append({'operation':'pivot','index':idx,'columns':cols,'values':vals,'aggfunc':agg,'description':'pivot'})
    elif op == 'melt':
        idv = st.multiselect('ID vars', df_orig.columns.tolist())
        valv = st.multiselect('Value vars', df_orig.columns.tolist())
        if st.button('Add'): st.session_state.steps.append({'operation':'melt','id_vars':idv,'value_vars':valv,'description':'melt'})
    elif op == 'trim':
        cols = st.multiselect('Columns', df_orig.select_dtypes(include='object').columns.tolist())
        if st.button('Add'): st.session_state.steps.append({'operation':'trim','columns':cols,'description':f"trim {cols}"})
    elif op == 'case':
        cols = st.multiselect('Columns', df_orig.select_dtypes(include='object').columns.tolist())
        case = st.selectbox('Case', ['lower','upper','title'])
        if st.button('Add'): st.session_state.steps.append({'operation':'case','columns':cols,'case':case,'description':f"case {case}"})
    elif op == 'dedupe':
        cols = st.multiselect('Subset (optional)', df_orig.columns.tolist())
        keep = st.selectbox('Keep', ['first','last','none'])
        if st.button('Add'): st.session_state.steps.append({'operation':'dedupe','subset':cols,'keep':keep,'description':'dedupe'})
    elif op == 'impute':
        cols = st.multiselect('Columns', df_orig.select_dtypes(include='number').columns.tolist())
        strat = st.selectbox('Strategy', ['mean','median','mode','constant'])
        const = None
        if strat == 'constant': const = st.text_input('Constant value')
        if st.button('Add'): st.session_state.steps.append({'operation':'impute','columns':cols,'strategy':strat,'constant':const,'description':'impute'})
    elif op == 'parse_date':
        cols = st.multiselect('Columns', [c for c in df_orig.columns if df_orig[c].dtype == 'object'])
        fmt = st.text_input('Format', value='%Y-%m-%d')
        if st.button('Add'): st.session_state.steps.append({'operation':'parse_date','columns':cols,'format':fmt,'description':'parse_date'})
    elif op == 'remove_special':
        cols = st.multiselect('Columns', df_orig.select_dtypes(include='object').columns.tolist())
        if st.button('Add'): st.session_state.steps.append({'operation':'remove_special','columns':cols,'description':'remove_special'})
    elif op == 'outlier':
        cols = st.multiselect('Columns', df_orig.select_dtypes(include='number').columns.tolist())
        thr = st.number_input('Z-score threshold', value=3.0)
        if st.button('Add'): st.session_state.steps.append({'operation':'outlier','columns':cols,'threshold':thr,'description':'outlier'})
    elif op == 'auto_clean':
        if st.button('Add Clean'): st.session_state.steps.append({'operation':'auto_clean','description':'auto_clean'})
    if st.button('Clear All'): st.session_state.steps.clear(); st.experimental_rerun()

# Apply and display
df_transformed = apply_steps(df_orig)
col1, col2 = st.columns(2)
with col1:
    if st.checkbox('Show Original'): st.dataframe(df_orig)
with col2:
    if st.checkbox('Show Transformed'): st.dataframe(df_transformed)

# Download
tab = st.tabs(['Download','Load & Access'])[1]
with st.tab('Download'):
    csv = df_transformed.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', csv, file_name=f'transformed_{sheet}.csv')
    if uploaded.name.lower().endswith(('.xls','.xlsx')):
        out = BytesIO();
        with pd.ExcelWriter(out, engine='xlsxwriter') as w: df_transformed.to_excel(w, sheet_name=sheet, index=False); w.save()
        st.download_button('Download Excel', out.getvalue(), file_name=f'transformed_{sheet}.xlsx')

with st.tab('Load & Access'):
    db_url = st.text_input('Database URL', value=os.getenv('DATABASE_URL',''))
    if st.button('Connect to DB'):
        try:
            eng = create_engine(db_url)
            st.session_state.db_engine = eng
            st.success('Connected')
        except Exception as e: st.error(e)
    if st.session_state.db_engine:
        eng = st.session_state.db_engine
        insp = inspect(eng)
        tables = insp.get_table_names()
        tbl = st.selectbox('Existing Tables', tables)
        if st.button('Load Table'):
            df_db = pd.read_sql_table(tbl, eng)
            st.dataframe(df_db)
        new_tbl = st.text_input('Save as Table', value=f"transformed_{sheet}" )
        if st.button('Save to DB'):
            try:
                df_transformed.to_sql(new_tbl, eng, if_exists='replace', index=False)
                st.success(f'Saved as {new_tbl}')
            except Exception as e: st.error(e)
