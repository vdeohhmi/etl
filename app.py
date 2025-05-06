import os
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sqlalchemy import create_engine
from pandasql import sqldf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import plotly.express as px
from datetime import datetime

# --- App Configuration ---
st.set_page_config(page_title="Data Transformer Pro Plus +", layout="wide")
st.title("🛠️ Data Transformer Pro Plus + 🔄 Snowflake")

# --- Session State Initialization ---
for key, default in [('df', None), ('df_aux', None), ('steps', []), ('versions', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Snowflake Connection Helper ---
def get_sf_conn():
    return snowflake.connector.connect(
        user=st.session_state.sf_user,
        password=st.session_state.sf_password,
        account=st.session_state.sf_account,
        warehouse=st.session_state.sf_warehouse,
        database=st.session_state.sf_database,
        schema=st.session_state.sf_schema
    )

# --- File and Snowflake Loaders ---
def load_file(uploaded):
    """Load CSV, Excel, Parquet, or JSON files."""
    ext = uploaded.name.split('.')[-1].lower()
    try:
        if ext == 'csv': return pd.read_csv(uploaded)
        if ext in ['xls', 'xlsx']: return pd.read_excel(uploaded, sheet_name=None)
        if ext == 'parquet': return pd.read_parquet(uploaded)
        if ext == 'json': return pd.read_json(uploaded)
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None

def load_snowflake():
    """Connect and load data from Snowflake using table name or SQL query."""
    with st.form("sf_conn_form", clear_on_submit=False):
        st.text_input("Snowflake Account", key='sf_account', help="Your Snowflake account identifier.")
        st.text_input("Username", key='sf_user', help="Snowflake user with data access.")
        st.text_input("Password", type='password', key='sf_password', help="Password for the Snowflake user.")
        st.text_input("Warehouse", key='sf_warehouse', help="Warehouse for compute resources.")
        st.text_input("Database", key='sf_database', help="Target database name.")
        st.text_input("Schema", key='sf_schema', help="Target schema name.")
        table = st.text_input("Table or SQL (prefix with SQL:)", key='sf_table', help="Enter a table name or a SQL query prefixed with SQL:.")
        submitted = st.form_submit_button("Load from Snowflake")
    if submitted:
        try:
            conn = get_sf_conn()
            if st.session_state.sf_table.strip().lower().startswith('sql:'):
                query = st.session_state.sf_table[4:].strip()
                df = pd.read_sql(query, conn)
            else:
                df = pd.read_sql(f"SELECT * FROM {st.session_state.sf_table}", conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Snowflake load failed: {e}")
    return None

# --- Transformation Function ---
def apply_steps(df):
    """Apply stored transformation steps to the DataFrame."""
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter':
            try:
                df = df.query(step['expr'])
            except Exception:
                st.warning(f"Invalid filter: {step['expr']}")
        elif t == 'compute':
            expr = step['expr']
            if expr.lower().startswith('sql:'):
                q = expr[4:].strip()
                try:
                    df = sqldf(q, {'df': df})
                except Exception:
                    st.warning(f"Invalid SQL: {q}")
            else:
                try:
                    df[step['new']] = df.eval(expr)
                except Exception:
                    st.warning(f"Invalid formula: {expr}")
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step['cols'])
        elif t == 'join' and st.session_state.df_aux is not None:
            df = df.merge(
                st.session_state.df_aux,
                left_on=step['left'],
                right_on=step['right'],
                how=step['how']
            )
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c].fillna(df[c].median(), inplace=True)
                    else:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
    return df

# --- UI Tabs ---
tabs = st.tabs([
    "📥 Load",
    "🔧 Transform",
    "📊 Profile",
    "💡 Insights",
    "⬇️ Export",
    "🕒 History"
])

# --- Load Tab ---
with tabs[0]:
    st.subheader("1. Load Data")
    source = st.radio(
        "Data Source", ['File', 'Snowflake'],
        help="Choose to load data from a local file or directly from Snowflake."
    )
    if source == 'File':
        uploaded = st.file_uploader(
            "Upload file (CSV, Excel, Parquet, JSON)",
            help="Drag and drop or browse to select a file."
        )
        if uploaded:
            data = load_file(uploaded)
            if isinstance(data, dict):
                sheet = st.selectbox(
                    "Select sheet", list(data.keys()),
                    help="For Excel files, choose which sheet to load."
                )
                data = data[sheet]
            if data is not None:
                st.session_state.df = data
                st.success("File loaded successfully.")
    else:
        df_sf = load_snowflake()
        if df_sf is not None:
            st.session_state.df = df_sf
            st.success("Data loaded from Snowflake.")
    if st.session_state.df is not None:
        st.data_editor(
            st.session_state.df,
            use_container_width=True,
            num_rows="dynamic"
        )

# --- Transform Tab ---
with tabs[1]:
    st.subheader("2. Build Transformations")
    df = st.session_state.df
    if df is None:
        st.info("Please load data first in the Load tab.")
    else:
        st.write("**Current Steps:**")
        for i, step in enumerate(st.session_state.steps):
            st.write(f"{i+1}. {step['type']} — {step.get('desc','')} ")
        st.markdown("---")
        op = st.selectbox(
            "Operation",
            ['rename','filter','compute','drop_const','onehot','join','impute'],
            help="Select the type of transformation to add."
        )
        if op == 'rename':
            old = st.selectbox(
                "Old column name", df.columns,
                help="Column to rename."        
            )
            new = st.text_input(
                "New column name",
                help="Enter the new name for the column."
            )
            if st.button("Add Rename"): st.session_state.steps.append({
                'type':'rename','old':old,'new':new,'desc':f"Rename {old} to {new}"
            })
        if op == 'filter':
            st.write("Available columns:", df.columns.tolist())
            expr = st.text_input(
                "Filter expression",
                help="Use pandas query syntax, e.g., `age > 30 and gender == 'M'`."
            )
            if st.button("Add Filter"): st.session_state.steps.append({
                'type':'filter','expr':expr,'desc':f"Filter: {expr}"
            })
        if op == 'compute':
            newc = st.text_input(
                "New column name", help="Name for the computed column."
            )
            expr2 = st.text_input(
                "Formula or SQL (prefix SQL:)",
                help="Enter a pandas eval formula or prefix a SQL query with `SQL:`."
            )
            if st.button("Add Compute"): st.session_state.steps.append({
                'type':'compute','new':newc,'expr':expr2,'desc':f"Compute {newc} = {expr2}"
            })
        if op == 'drop_const' and st.button("Add Drop Constants", help="Remove columns with constant values." ):
            st.session_state.steps.append({'type':'drop_const','desc':'Drop constant columns'})
        if op == 'onehot':
            cols = st.multiselect(
                "Columns to one-hot encode", df.select_dtypes('object').columns,
                help="Select categorical columns to expand into dummy variables."
            )
            if st.button("Add One-Hot"): st.session_state.steps.append({
                'type':'onehot','cols':cols,'desc':f"One-hot encode {cols}"
            })
        if op == 'join' and st.session_state.df_aux is not None:
            left = st.selectbox(
                "Left key column", df.columns,
                help="Column in the primary dataset for the join."
            )
            right = st.selectbox(
                "Right key column", st.session_state.df_aux.columns,
                help="Column in the auxiliary dataset for the join."
            )
            how = st.selectbox(
                "Join type", ['inner','left','right','outer'],
                help="Type of SQL-style join to perform."
            )
            if st.button("Add Join"): st.session_state.steps.append({
                'type':'join','left':left,'right':right,'how':how,'desc':f"Join {how} on {left}={right}"
            })
        if op == 'impute' and st.button("Add Auto-Impute", help="Fill missing values with median/mode." ):
            st.session_state.steps.append({'type':'impute','desc':'Auto-impute missing values'})
        if st.button("Apply Transformations", help="Execute all added transformation steps."):
            st.session_state.df = apply_steps(df)
            st.success("Transformations applied.")
        st.data_editor(
            st.session_state.df,
            use_container_width=True,
            num_rows="dynamic"
        )

# --- Profile Tab ---
with tabs[2]:
    st.subheader("3. Data Profile")
    df = st.session_state.df
    if df is not None:
        profile = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'null_pct': df.isna().mean()*100
        })
        st.dataframe(profile, help="Overview of column types and missingness.")

# --- Insights Tab ---
with tabs[3]:
    st.subheader("4. Auto Insights")
    df = st.session_state.df
    if df is not None:
        num = df.select_dtypes('number')
        if not num.empty:
            fig = px.imshow(
                num.corr(), text_auto=True, title='Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True, help="Heatmap of numeric column correlations.")
        miss = df.isna()
        fig2 = px.imshow(
            miss, title='Missingness Heatmap'
        )
        st.plotly_chart(fig2, use_container_width=True, help="Visualize missing data patterns.")

# --- Export Tab ---
with tabs[4]:
    st.subheader("5. Export & Writeback")
    df = st.session_state.df
    if df is not None:
        fmt = st.selectbox(
            "Select export format",
            ['CSV','JSON','Parquet','Excel','Snowflake'],
            help="Choose file format or write back to Snowflake."
        )
        if st.button("Execute Export", help="Download or writeback the current DataFrame."):
            if fmt == 'CSV':
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode(),
                    "data.csv"
                )
            elif fmt == 'JSON':
                st.download_button(
                    "Download JSON",
                    df.to_json(orient='records'),
                    "data.json"
                )
            elif fmt == 'Parquet':
                st.download_button(
                    "Download Parquet",
                    df.to_parquet(index=False),
                    "data.parquet"
                )
            elif fmt == 'Excel':
                out = BytesIO()
                df.to_excel(out, index=False, engine='openpyxl')
                st.download_button(
                    "Download Excel",
                    out.getvalue(),
                    "data.xlsx"
                )
            else:
                with st.form("sf_writeback", clear_on_submit=False):
                    tbl = st.text_input(
                        "Snowflake table name",
                        help="Existing or new table to write data into."
                    )
                    submitted = st.form_submit_button("Write to Snowflake")
                if submitted:
                    try:
                        conn = get_sf_conn()
                        write_pandas(conn, df, tbl)
                        st.success(f"Written back to Snowflake table {tbl}")
                        conn.close()
                    except Exception as e:
                        st.error(f"Snowflake writeback failed: {e}")

# --- History Tab ---
with tabs[5]:
    st.subheader("6. Transformation History")
    for idx,(ts,snap) in enumerate(st.session_state.versions):
        cols = st.columns([0.7, 0.3])
        cols[0].write(f"{idx+1}. {ts}")
        if cols[1].button("Revert", key=f"rev{idx}"):
            st.session_state.df = snap
            st.experimental_rerun()
