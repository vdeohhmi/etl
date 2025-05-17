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

# --- OpenAI API Key (from ENV) ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Set the OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key)

# --- Helper to clean AI expressions ---
def clean_expr(raw: str) -> str:
    lines = raw.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].lstrip().startswith("```"):
        lines = lines[:-1]
    expr = "\n".join(lines)
    return expr.replace("`", "").strip()

# --- Helper Functions ---
def load_file(f):
    ext = f.name.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(f)
        if ext in ['xls','xlsx']:
            return pd.read_excel(f, sheet_name=None)
        if ext == 'parquet':
            return pd.read_parquet(f)
        if ext == 'json':
            return pd.read_json(f)
    except Exception as e:
        st.error(f"Failed to load {f.name}: {e}")
    return None


def apply_steps(df: pd.DataFrame) -> pd.DataFrame:
    st.session_state.versions.append(df.copy())
    for s in st.session_state.steps:
        t = s.get('type')
        if t == 'rename':
            df = df.rename(columns={s['old']: s['new']})
        elif t == 'filter' and s.get('expr'):
            try:
                df = df.query(s['expr'])
            except:
                pass
        elif t == 'compute' and s.get('expr'):
            expr = s['expr']
            try:
                df[s['new']] = df.eval(expr)
            except:
                df[s['new']] = df.eval(expr, engine='python')
        elif t == 'drop_const':
            df = df.loc[:, df.nunique() > 1]
        elif t == 'onehot':
            df = pd.get_dummies(df, columns=s.get('cols', []))
        elif t == 'join':
            aux = st.session_state.datasets.get(s['aux'])
            if aux is not None:
                df = df.merge(
                    aux,
                    left_on=s['left'],
                    right_on=s['right'],
                    how=s.get('how','inner')
                )
        elif t == 'impute':
            for c in df.columns:
                if df[c].isna().any():
                    df[c] = df[c].fillna(
                        df[c].median() if pd.api.types.is_numeric_dtype(df[c])
                        else df[c].mode().iloc[0]
                    )
    return df

# --- UI Tabs ---
tabs = st.tabs([
    '📂 Datasets',
    '✏️ Transform',
    '📈 Profile',
    '💡 Insights',
    '⬇️ Export',
    '🕒 History',
    '⚙️ Snowflake',
    '🤖 AI Toolkit',
    '🕸️ Social Graph'
])

# 1. Datasets
with tabs[0]:
    st.header('1. Datasets')
    uploads = st.file_uploader(
        'Upload files (CSV/Excel/Parquet/JSON)',
        type=['csv','xls','xlsx','parquet','json'],
        accept_multiple_files=True
    )
    if uploads:
        for f in uploads:
            data = load_file(f)
            if isinstance(data, dict):
                for sheet, sdf in data.items():
                    st.session_state.datasets[f'{f.name}:{sheet}'] = sdf
            elif data is not None:
                st.session_state.datasets[f.name] = data
        st.success('Datasets loaded.')
    if st.session_state.datasets:
        key = st.selectbox('Select dataset', list(st.session_state.datasets.keys()))
        st.session_state.current = key
        st.data_editor(
            st.session_state.datasets[key],
            key=f'editor_{key}',
            use_container_width=True
        )

# 2. Transform
with tabs[1]:
    st.header('2. Transform')
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        st.write('**Steps:**')
        for i, step in enumerate(st.session_state.steps, 1):
            st.write(f'{i}. {step["type"]} — {step.get("desc","")})')
        op = st.selectbox('Operation', ['rename','filter','compute','drop_const','onehot','join','impute'])
        with st.form('transform_form'):
            if op == 'rename':
                old = st.selectbox('Old column', df.columns)
                new = st.text_input('New column name')
                if st.form_submit_button('Add Rename'):
                    st.session_state.steps.append({'type':'rename','old':old,'new':new,'desc':f'{old}→{new}'})
            elif op == 'filter':
                expr = st.text_input('Filter expression')
                if st.form_submit_button('Add Filter'):
                    st.session_state.steps.append({'type':'filter','expr':expr,'desc':expr})
            elif op == 'compute':
                newc = st.text_input('New column name')
                desc = st.text_area('Describe logic')
                if st.form_submit_button('AI Generate & Add Compute'):
                    prompt = f"Generate pandas expression for {newc}: {desc}"
                    resp = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[{'role':'user','content':prompt}]
                    )
                    expr = clean_expr(resp.choices[0].message.content)
                    st.code(expr)
                    st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
            elif op == 'drop_const':
                if st.form_submit_button('Add Drop Constants'):
                    st.session_state.steps.append({'type':'drop_const','desc':'drop constants'})
            elif op == 'onehot':
                cols = st.multiselect('Columns to encode', df.select_dtypes('object').columns)
                if st.form_submit_button('Add One-Hot'):
                    st.session_state.steps.append({'type':'onehot','cols':cols,'desc':','.join(cols)})
            elif op == 'join':
                aux = st.selectbox('Aux dataset',[k for k in st.session_state.datasets if k!=key])
                left = st.selectbox('Left key', df.columns)
                right = st.selectbox('Right key', st.session_state.datasets[aux].columns)
                how = st.selectbox('Join type', ['inner','left','right','outer'])
                if st.form_submit_button('Add Join'):
                    st.session_state.steps.append({'type':'join','aux':aux,'left':left,'right':right,'how':how,'desc':f'{aux} on {left}={right}'})
            else:
                if st.form_submit_button('Add Impute'):
                    st.session_state.steps.append({'type':'impute','desc':'impute'})
        updated = apply_steps(df)
        st.session_state.datasets[key] = updated
        st.write('**Preview:**')
        st.data_editor(
            updated,
            key=f'preview_{key}_{len(st.session_state.versions)}',
            use_container_width=True
        )

# 3. Profile
with tabs[2]:
    st.header('3. Profile')
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        stats = pd.DataFrame({
            'dtype':df.dtypes,
            'nulls':df.isna().sum(),
            'null_pct':df.isna().mean()*100
        })
        st.dataframe(stats, use_container_width=True)

# 4. Insights
with tabs[3]:
    st.header('4. Insights')
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        num = df.select_dtypes('number')
        if not num.empty:
            st.plotly_chart(px.imshow(num.corr(), text_auto=True), use_container_width=True)

# 5. Export
with tabs[4]:
    st.header('5. Export')
    key = st.session_state.current
    if key:
        df = st.session_state.datasets[key]
        fmt = st.selectbox('Format',['CSV','JSON','Parquet','Excel','Snowflake'])
        if st.button('Export'):
            if fmt=='CSV':
                st.download_button('CSV', df.to_csv(index=False).encode(), 'data.csv')
            elif fmt=='JSON':
                st.download_button('JSON', df.to_json(orient='records'), 'data.json')
            elif fmt=='Parquet':
                st.download_button('Parquet', df.to_parquet(index=False), 'data.parquet')
            elif fmt=='Excel':
                buf=BytesIO(); df.to_excel(buf,index=False,engine='openpyxl'); st.download_button('Excel', buf.getvalue(), 'data.xlsx')
            else:
                tbl=st.text_input('Snowflake table name')
                if st.button('Write to Snowflake'):
                    conn=snowflake.connector.connect(
                        user=st.session_state['sf_user'],
                        password=st.session_state['sf_password'],
                        account=st.session_state['sf_account'],
                        warehouse=st.session_state['sf_warehouse'],
                        database=st.session_state['sf_database'],
                        schema=st.session_state['sf_schema']
                    )
                    write_pandas(conn,df,tbl)
                    conn.close()
                    st.success(f'Wrote to {tbl}')

# 6. History
with tabs[5]:
    st.header('6. History')
    key = st.session_state.current
    if key and st.session_state.versions:
        for i,snap in enumerate(st.session_state.versions,1):
            c1,c2=st.columns([0.7,0.3])
            c1.write(f'{i}. Snapshot')
            if c2.button('Revert', key=f'hist_{i}'):
                st.session_state.datasets[key]=snap
        st.data_editor(
            st.session_state.datasets[key],
            key=f'hist_prev_{len(st.session_state.versions)}',
            use_container_width=True
        )

# 7. Snowflake Settings
with tabs[6]:
    st.header('7. Snowflake Settings')
    st.text_input('Account', key='sf_account')
    st.text_input('User', key='sf_user')
    st.text_input('Password', type='password', key='sf_password')
    st.text_input('Warehouse', key='sf_warehouse')
    st.text_input('Database', key='sf_database')
    st.text_input('Schema', key='sf_schema')

# 8. AI Toolkit
with tabs[7]:
    st.header('8. AI Toolkit')
    key = st.session_state.current
    if not key:
        st.info('Select a dataset')
    else:
        df = st.session_state.datasets[key]
        st.data_editor(df, key=f'ai_prev_{key}_{len(st.session_state.steps)}', use_container_width=True)
        tool = st.selectbox('Tool',['Compute Column','NL Query','Data Story'])
        if tool=='Compute Column':
            newc=st.text_input('New col name', key='ai_newc')
            desc=st.text_area('Logic', key='ai_desc')
            with st.form('ai_form'):
                if st.form_submit_button('Gen & Apply'):
                    prompt=f"Expr for {newc}: {desc}"
                    resp=client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}])
                    expr=clean_expr(resp.choices[0].message.content)
                    st.code(expr)
                    st.session_state.steps.append({'type':'compute','new':newc,'expr':expr,'desc':desc})
            alt=st.text_input('Or paste formula', key='ai_alt')
            if st.button('Apply Alt'):
                st.session_state.steps.append({'type':'compute','new':newc,'expr':alt,'desc':'manual'})
            df2=apply_steps(df)
            st.session_state.datasets[key]=df2
            st.data_editor(df2, key=f'ai_upd_{key}_{len(st.session_state.steps)}', use_container_width=True)
        elif tool=='NL Query':
            q=st.text_area('Question', key='ai_q')
            if st.button('Run'): resp=client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':f'Q:{q}'}]); st.markdown(resp.choices[0].message.content)
        else:
            mode=st.selectbox('Story for',['Entire Dataset','Single Column'])
            if mode=='Single Column':
                col=st.selectbox('Col', df.columns, key='ai_story_col')
                if st.button('Gen Story'): resp=client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':f'Analyze {col}'}]); st.markdown(resp.choices[0].message.content)
            else:
                if st.button('Gen DS Story'): resp=client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'Report dataset'}]); st.markdown(resp.choices[0].message.content)

# 9. Social Graph
with tabs[8]:
    st.header('9. Social Graph')
    key=st.session_state.current
    if key:
        df=st.session_state.datasets[key]
        src=st.selectbox('Source', df.columns, key='sg_src')
        tgt=st.selectbox('Target', df.columns, key='sg_tgt')
        wt_opt=[None]+list(df.columns)
        wt=st.selectbox('Weight', wt_opt, key='sg_wt')
        if st.button('Gen Graph'):
            G=nx.Graph()
            for _,r in df.iterrows():
                u,v=r[src],r[tgt]; w=float(r[wt]) if wt and pd.notna(r[wt]) else 1.0
                if G.has_edge(u,v): G[u][v]['weight']+=w
                else: G.add_edge(u,v,weight=w)
            top5=set(sorted(G.edges(data=True), key=lambda x:x[2]['weight'], reverse=True)[:5])
            net=Network(height='700px',width='100%',bgcolor='#222222',font_color='white'); net.show_buttons(filter_=['physics'])
            for n in G.nodes(): net.add_node(n,label=str(n),title=f'Deg:{G.degree(n)}',value=G.degree(n))
            for u,v,d in G.edges(data=True): net.add_edge(u,v,value=d['weight'],width=4 if (u,v) in top5 else 1,color='red' if (u,v) in top5 else 'rgba(200,200,200,0.2)',title=f"W:{d['weight']}")
            import streamlit.components.v1 as components; components.html(net.generate_html(),height=750,scrolling=True)
