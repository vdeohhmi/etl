import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine

st.set_page_config(page_title="ETL Tool", layout="wide")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def fetch_api(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return pd.json_normalize(data)

@st.cache_data
def query_db(conn_str, query):
    engine = create_engine(conn_str)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


def extract_data():
    st.sidebar.header("Extraction")
    method = st.sidebar.radio("Select method", ["CSV Upload", "API Endpoint", "Database Query"])
    df = None
    if method == "CSV Upload":
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"]);
        if file: df = load_csv(file)
    elif method == "API Endpoint":
        url = st.sidebar.text_input("API URL")
        hdrs = st.sidebar.text_area("Headers (JSON)", "{}")
        if st.sidebar.button("Fetch API Data"):
            headers = eval(hdrs)
            df = fetch_api(url, headers)
    else:
        conn_str = st.sidebar.text_input("DB Connection (SQLAlchemy)")
        query = st.sidebar.text_area("SQL Query")
        if st.sidebar.button("Run Query"):
            df = query_db(conn_str, query)
    return df


def transform_data(df):
    if df is None:
        return None
    st.header("Transformation")
    st.dataframe(df.head())

    if st.expander("Rename Columns"):
        mapping = st.text_area("Mapping as Python dict", "{}")
        if st.button("Apply Rename"):
            df.rename(columns=eval(mapping), inplace=True)
            st.success("Columns renamed")

    if st.expander("Filter Rows"):
        expr = st.text_input("Pandas query expression", "")
        if st.button("Apply Filter") and expr:
            df = df.query(expr)
            st.success("Filter applied")

    if st.expander("Add Computed Column"):
        new_col = st.text_input("Column name")
        expr = st.text_area("Expression (pandas eval)")
        if st.button("Add Column") and new_col and expr:
            df[new_col] = df.eval(expr)
            st.success(f"Column '{new_col}' added")

    st.write(df)
    return df


def load_data(df):
    if df is None:
        return
    st.header("Load")
    choice = st.selectbox("Load to", ["Download CSV", "Database"])
    if choice == "Download CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "data_export.csv", "text/csv")
    else:
        db_url = st.text_input("DB Connection (SQLAlchemy)")
        tbl = st.text_input("Target Table Name")
        if st.button("Write to DB"):
            engine = create_engine(db_url)
            df.to_sql(tbl, engine, if_exists='replace', index=False)
            engine.dispose()
            st.success(f"Data written to table '{tbl}'")


def main():
    st.title("ETL Tool Web App")
    df = extract_data()
    df = transform_data(df)
    load_data(df)


if __name__ == "__main__":
    main()
