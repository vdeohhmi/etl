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
    ("datasets", {}),
    ("current", None),
    ("steps", []),
    ("versions", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- OpenAI API Key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# --- Helpers ---
def clean_expr(raw: str) -> str:
    # Remove Markdown fences/backticks
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    expr = "\n".join(lines)
    return expr.replace("`", "").strip()


def load_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(uploaded_file)
        elif ext in ('xls', 'xlsx'):
            return pd.read_excel(uploaded_file, sheet_name=None)
        elif ext == 'parquet':
            return pd.read_parquet(uploaded_file)
        elif ext == 'json':
            return pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {e}")
    return None


def apply_steps(df: pd.DataFrame) -> pd.DataFrame:
    # Snapshot
    st.session_state.versions.append(df.copy())
    for step in st.session_state.steps:
        t = step['type']
        if t == 'rename':
            df = df.rename(columns={step['old']: step['new']})
        elif t == 'filter' and step.get('expr'):
            try:
                df = df.query(step['expr'])
            except Exception:
                pass
        elif t == 'compute' and step.get('expr'):
            expr = step['expr']
            try:
                df[step['new']] = df.eval(expr)
            except Exception:
                df[step['new']] = df.eval(expr, engine='python')
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=step.get('cols', []))
        elif t == 'join':
            aux_df = st.session_state.datasets.get(step['aux'])
            if aux_df is not None:
                df = df.merge(aux_df,
                              left_on=step['left'],
                              right_on=step['right'],
                              how=step.get('how', 'inner'))
        elif t == 'impute':
            for col in df.columns:
                if df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# --- UI Tabs ---
(tab_datasets,
 tab_transform,
 tab_profile,
 tab_insights,
 tab_export,
 tab_history,
 tab_snowflake,
 tab_ai,
 tab_graph) = st.tabs([
    '📂 Datasets', '✏️ Transform', '📈 Profile', '💡 Insights',
    '⬇️ Export', '🕒 History', '⚙️ Snowflake', '🤖 AI Toolkit', '🕸️ Social Graph'
])

# 1. Datasets
with tab_datasets:
    st.header("1. Datasets")
    files = st.file_uploader(
        "Upload CSV/Excel/Parquet/JSON",
        type=['csv','xls','xlsx','parquet','json'],
        accept_multiple_files=True
    )
    if files:
        for f in files:
            data = load_file(f)
            if isinstance(data, dict):
                for sheet, df_sheet in data.items():
                    st.session_state.datasets[f"{f.name}:{sheet}"] = df_sheet
            elif isinstance(data, pd.DataFrame):
                st.session_state.datasets[f.name] = data
        st.success("Loaded datasets.")
    if st.session_state.datasets:
        choice = st.selectbox(
            "Select a dataset", list(st.session_state.datasets.keys())
        )
        st.session_state.current = choice
        st.data_editor(
            st.session_state.datasets[choice],
            key=f"editor_{choice}", use_container_width=True
        )

# 2. Transform
with tab_transform:
    st.header("2. Transform")
    key = st.session_state.current
    if key and key in st.session_state.datasets:
        df = st.session_state.datasets[key]
        st.write("**Steps:**")
        for i, step in enumerate(st.session_state.steps, 1):
            st.write(f"{i}. {step['type']} — {step.get('desc','')}")
        operation = st.selectbox(
            "Operation", ['rename','filter','compute','drop_const','onehot','join','impute']
        )
        with st.form("transform_form"):
            if operation == 'rename':
                old = st.selectbox("Old column", df.columns)
                new = st.text_input("New column name")
                submitted = st.form_submit_button("Add Rename")
                if submitted:
                    st.session_state.steps.append({
                        'type':'rename','old':old,'new':new,
                        'desc':f"{old} → {new}"
                    })
            elif operation == 'filter':
                expr = st.text_input("Filter expression (pandas query)")
                submitted = st.form_submit_button("Add Filter")
                if submitted:
                    st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
            elif operation == 'compute':
                new_col = st.text_input("New column name")
                logic = st.text_area("Logic (plain English)")
                submitted = st.form_submit_button("AI Generate & Add Compute")
                if submitted:
                    cols = df.columns.tolist()
                    sample = df.head(3).to_dict('records')
                    prompt = (
                        f"You are a Python data engineer. Columns: {cols}. "
                        f"Sample: {sample}. Generate a pandas eval expression "
                        f"for '{new_col}' with logic: {logic}. Return the expression."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{'role':'user','content':prompt}]
                    )
                    expr = clean_expr(resp.choices[0].message.content)
                    st.code(expr)
                    st.session_state.steps.append({
                        'type':'compute','new':new_col,'expr':expr,'desc':logic
                    })
            elif operation == 'drop_const':
                if st.form_submit_button("Add Drop Constants"):
                    st.session_state.steps.append({'type':'drop_const','desc':'drop constant columns'})
            elif operation == 'onehot':
                cols = st.multiselect("Columns to one-hot encode", df.select_dtypes('object').columns)
                submitted = st.form_submit_button("Add One-Hot")
                if submitted:
                    st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
            elif operation == 'join':
                aux = st.selectbox("Aux dataset", [d for d in st.session_state.datasets if d!=key])
                left = st.selectbox("Left key", df.columns)
                right = st.selectbox("Right key", st.session_state.datasets[aux].columns)
                how = st.selectbox("Join type", ['inner','left','right','outer'])
                submitted = st.form_submit_button("Add Join")
                if submitted:
                    st.session_state.steps.append({
                        'type':'join','aux':aux,'left':left,'right':right,'how':how,
                        'desc':f"join {aux} on {left}={right}"
                    })
            elif operation == 'impute':
                if st.form_submit_button("Add Impute"):
                    st.session_state.steps.append({'type':'impute','desc':'auto-impute'})
        # Apply and preview
        result = apply_steps(df)
        st.session_state.datasets[key] = result
        st.write("**Preview after Transform:**")
        st.data_editor(result, key=f"preview_{key}_{len(st.session_state.versions)}", use_container_width=True)

# 3. Profile
with tab_profile:
    st.header("3. Profile")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({
            'dtype': df.dtypes,
            'nulls': df.isna().sum(),
            'null_pct': df.isna().mean()*100
        })
        st.dataframe(stats, use_container_width=True)

# 4. Insights
with tab_insights:
    st.header("4. Insights")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes('number')
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# 5. Export
with tab_export:
    st.header("5. Export")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox('Choose format', ['CSV','JSON','Parquet','Excel','Snowflake'])
        if st.button('Export'):
            if fmt=='CSV':
                st.download_button('Download CSV', df.to_csv(index=False).encode(), 'data.csv')
            elif fmt=='JSON':
                st.download_button('Download JSON', df.to_json(orient='records'), 'data.json')
            elif fmt=='Parquet':
                st.download_button('Download Parquet', df.to_parquet(index=False), 'data.parquet')
            elif fmt=='Excel':
                buf = BytesIO(); df.to_excel(buf, index=False, engine='openpyxl');
                st.download_button('Download Excel', buf.getvalue(), 'data.xlsx')
            else:
                tbl = st.text_input('Snowflake table name')
                if st.button('Write to Snowflake'):
                    conn = snowflake.connector.connect(
                        user=st.session_state['sf_user'],
                        password=st.session_state['sf_password'],
                        account=st.session_state['sf_account'],
                        warehouse=st.session_state['sf_warehouse'],
                        database=st.session_state['sf_database'],
                        schema=st.session_state['sf_schema']
                    )
                    write_pandas(conn, df, tbl)
                    conn.close()
                    st.success(f'Wrote to {tbl}')

# 6. History
with tab_history:
    st.header("6. History")
    key = st.session_state.current
    if key and st.session_state.versions:
        for i, snap in enumerate(st.session_state.versions, 1):
            c1, c2 = st.columns([0.7, 0.3])
            c1.write(f'{i}. Snapshot')
            if c2.button('Revert', key=f'hist_revert_{i}'):
                st.session_state.datasets[key] = snap
        st.data_editor(st.session_state.datasets[key], key=f'hist_prev_{len(st.session_state.versions)}', use_container_width=True)

# 7. Snowflake Settings
with tab_snowflake:
    st.header("7. Snowflake Settings")
    st.text_input('Account', key='sf_account')
    st.text_input('User', key='sf_user')
    st.text_input('Password', type='password', key='sf_password')
    st.text_input('Warehouse', key='sf_warehouse')
    st.text_input('Database', key='sf_database')
    st.text_input('Schema', key='sf_schema')

# 8. AI Toolkit
with tab_ai:
    st.header("8. AI Toolkit")
    key = st.session_state.current
    if not key:
        st.info('Load a dataset to use AI tools')
    else:
        df = st.session_state.datasets[key]
        st.data_editor(df, key=f'ai_preview_{key}_{len(st.session_state.steps)}', use_container_width=True)
        tool = st.selectbox('AI Tool', ['Compute Column','Natural Language Query','Data Storytelling'])
        if tool == 'Compute Column':
            newc = st.text_input('New column name', key='ai_newc')
            desc = st.text_area('Logic', key='ai_desc')
            with st.form('ai_compute_form'):
                submit = st.form_submit_button('Generate & Apply')
                if submit:
                    cols = df.columns.tolist()
                    sample = df.head(3).to_dict('records')
                    prompt = f"You are a Python data engineer. Columns: {cols}. Sample: {sample}. Generate pandas expression for '{newc}' logic: {desc}."
                    resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
                    expr = clean_expr(resp.choices[0].message.content)
                    st.code(expr)
                    st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
            alt = st.text_input('Or paste formula', key='ai_alt')
            if st.button('Apply Alternate') and alt.strip():
                st.session_state.steps.append({'type':'compute','new':newc,'expr':alt,'desc':'manual'})
            df2 = apply_steps(df)
            st.session_state.datasets[key] = df2
            st.data_editor(df2, key=f'ai_updated_{key}_{len(st.session_state.steps)}', use_container_width=True)
        # Natural Language Query
        elif tool == 'Natural Language Query':
            q = st.text_area('Ask a question', key='ai_query')
            if st.button('Run Query'):
                cols = df.columns.tolist()
                sample = df.head(5).to_dict('records')
                prompt = f"You are a data analyst. Columns: {cols}. Sample: {sample}. Question: {q}."  
                resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
                st.markdown(resp.choices[0].message.content)
        # Data Storytelling
        else:
            mode = st.selectbox('Story for', ['Entire Dataset','Single Column'], key='ai_story_mode')
            if mode == 'Single Column':
                col = st.selectbox('Column', df.columns, key='ai_story_col')
                if st.button('Generate Story'):
                    cols = df.columns.tolist()
                    sample = df.head(5).to_dict('records')
                    prompt = f"You are a data journalist. Columns: {cols}. Sample: {sample}. Analyze column '{col}'."
                    resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
                    st.markdown(resp.choices[0].message.content)
            else:
                if st.button('Generate Dataset Story'):
                    cols = df.columns.tolist()
                    sample = df.head(5).to_dict('records')
                    prompt = f"You are a data journalist. Columns: {cols}. Sample: {sample}. Write a detailed dataset report."
                    resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
                    st.markdown(resp.choices[0].message.content)

# 9. Social Graph
with tab_graph:
    st.header("9. Social Graph")
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        src = st.selectbox('Source', df.columns, key='sg_src')
        tgt = st.selectbox('Target', df.columns, key='sg_tgt')
        wt_opt = [None] + list(df.columns)
        wt = st.selectbox('Weight', wt_opt, key='sg_wt')
        if st.button('Generate Graph'):
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[src], row[tgt]
                w = float(row[wt]) if wt and pd.notnull(row[wt]) else 1.0
                G.add_edge(u, v, weight=G[u][v]['weight']+w if G.has_edge(u,v) else w)
            edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            top5 = { (u,v) for u,v,_ in edges[:5] }
            net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white')
            net.show_buttons(filter_=['physics'])
            for n in G.nodes():
                net.add_node(n, label=str(n), title=f'Degree: {G.degree(n)}', value=G.degree(n))
            for u,v,d in G.edges(data=True):
                width = 4 if (u,v) in top5 or (v,u) in top5 else 1
                color = 'red' if (u,v) in top5 or (v,u) in top5 else 'rgba(200,200,200,0.2)'
                net.add_edge(u, v, value=d['weight'], width=width, color=color, title=f"Weight: {d['weight']}")
            html = net.generate_html()
            import streamlit.components.v1 as components
            components.html(html, height=750, scrolling=True)
