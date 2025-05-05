```python
import os
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from sqlalchemy import create_engine
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# --- CONFIG & CACHING ---
st.set_page_config(page_title="Next-Gen ETL Orchestrator", layout="wide")
@st.experimental_singleton
def load_config(path='config.yaml'):
    return yaml.safe_load(open(path)) if os.path.exists(path) else {}

config = load_config()

# --- SLACK NOTIFICATIONS ---
slack_token = os.getenv('SLACK_TOKEN')
slack_client = WebClient(token=slack_token) if slack_token else None

def notify_slack(channel, message):
    if not slack_client: return
    try:
        slack_client.chat_postMessage(channel=channel, text=message)
    except SlackApiError as e:
        st.error(f"Slack notify failed: {e.response['error']}")

# --- ETL STEP FUNCTIONS ---
def extract(step):
    t = step['type']
    if t == 'csv': return pd.read_csv(step['path'])
    if t == 'api': return pd.json_normalize(requests.get(step['url'], headers=step.get('headers', {})).json())
    if t == 'db':
        eng = create_engine(step['connection'])
        df = pd.read_sql(step['query'], eng)
        eng.dispose(); return df
    raise ValueError(f"Invalid extract type: {t}")


def transform(df, step):
    if 'filter' in step:
        return df.query(step['filter'])
    if 'rename' in step:
        return df.rename(columns=step['rename'])
    if 'compute' in step:
        df[step['compute']['col']] = df.eval(step['compute']['expr'])
        return df
    if 'pivot' in step:
        return df.pivot_table(**step['pivot'])
    return df


def load(df, step):
    if step['type'] == 'csv':
        df.to_csv(step['path'], index=False)
    elif step['type'] == 'db':
        eng = create_engine(step['connection'])
        df.to_sql(step['table'], eng, if_exists=step.get('if_exists','append'), index=False)
        eng.dispose()

# --- PIPELINE EXECUTION ---
def run_pipeline(name):
    steps = config.get(name, {})
    df = None
    logs = []
    try:
        for s in steps.get('extract', []):
            df = extract(s); logs.append(f"Extracted: {s['type']}")
        for s in steps.get('transform', []):
            df = transform(df, s); logs.append(f"Transformed: {list(s.keys())[0]}")
        for s in steps.get('load', []):
            load(df, s); logs.append(f"Loaded: {s['type']}")
        logs.append("✅ Pipeline succeeded")
        notify_slack(steps.get('notify_channel',''), f"Pipeline {name} succeeded at {datetime.utcnow()}")
    except Exception as e:
        logs.append(f"❌ Error: {e}")
        notify_slack(steps.get('notify_channel',''), f"Pipeline {name} failed: {e}")
    return df, logs

# --- SCHEDULER SETUP ---
scheduler = BackgroundScheduler()
for name,p in config.items():
    cron = p.get('schedule')
    if cron:
        cron_fields = cron.split()
        scheduler.add_job(run_pipeline, 'cron', args=[name],
                          minute=cron_fields[0], hour=cron_fields[1],
                          day=cron_fields[2], month=cron_fields[3], day_of_week=cron_fields[4],
                          id=name)
scheduler.start()

# --- UI LAYOUT ---
st.title("🚀 Next-Gen ETL Orchestrator 🚀")
col1, col2 = st.columns([1,3])
with col1:
    st.sidebar.header("Pipelines")
    selected = st.sidebar.selectbox("Pipeline", list(config.keys()))
    mode = st.sidebar.radio("Mode", ['Preview','Run','Metrics','Logs','Schedule'])

with col2:
    if mode == 'Preview':
        df, _ = run_pipeline(selected)
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{selected}.csv")

    elif mode == 'Run':
        if st.button(f"Run {selected}"):
            with st.spinner("Executing..."):
                df, logs = run_pipeline(selected)
            for msg in logs: st.write(msg)

    elif mode == 'Metrics':
        df,_ = run_pipeline(selected)
        st.metric(label="Rows", value=df.shape[0])
        st.metric(label="Columns", value=df.shape[1])
        st.markdown("#### Column Distributions")
        for col in df.select_dtypes('number').columns:
            fig = px.histogram(df, x=col, title=col)
            st.plotly_chart(fig, use_container_width=True)

    elif mode == 'Logs':
        st.markdown("### Scheduler Jobs")
        jobs = scheduler.get_jobs()
        for job in jobs: st.write(f"• {job.id} → {job.next_run_time}")

    elif mode == 'Schedule':
        cron = st.text_input("Cron Expr", config[selected].get('schedule',''))
        chan = st.text_input("Slack Channel", config[selected].get('notify_channel',''))
        if st.button("Update Schedule"):
            config[selected]['schedule'], config[selected]['notify_channel'] = cron, chan
            yaml.safe_dump(config, open('config.yaml','w'))
            st.success("Schedule updated")

st.sidebar.markdown("---")
st.sidebar.markdown("Built-in Slack alerts, multi-step metrics, cron jobs, and pivot transforms!")
```
