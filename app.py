import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.express as px

# Streamlit page configuration
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.hdqwalls.com/wallpapers/dark-abstract-black-minimal-4k-q0.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.5);
    color: white;
}
[data-testid="stMarkdownContainer"] {
    color: #FFFFFF;
}
.title {
    color: #00BFFF; 
    font-size: 36px; 
    font-weight: bold;
}
.footer {
    color: #FFFFFF;
    font-size: 14px;
    text-align: center;
    padding: 10px;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <div class="title">
        Telecom EDA Web App
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander('Upload your Data', expanded=True):
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    def load_data(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            xls = pd.ExcelFile(file)
            sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
            return pd.read_excel(xls, sheet_name)
        else:
            st.error("Unsupported file format.")
            return None

    df = load_data(uploaded_file)

    if df is not None:
        st.subheader('DataFrame Preview')
        st.dataframe(df)
        st.write('---')

        analysis_type = st.selectbox("Select Analysis Type", [
            "Descriptive Statistics", 
            "Correlation Analysis", 
            "Time Series Analysis",
            "Logistic Regression",
            "Random Forest Classification",
            "Clustering (K-Means)",
            "ROC Curve",
            "F1 Score",
            "Network Graph Analysis"
        ])

        if analysis_type == "Descriptive Statistics":
            with st.expander('Descriptive Statistics', expanded=True):
                st.write(df.describe(include='all'))

        elif analysis_type == "Correlation Analysis":
            with st.expander('Correlation Analysis', expanded=True):
                corr_columns = st.multiselect("Select Columns for Correlation", df.columns)
                if corr_columns:
                    try:
                        corr_df = df[corr_columns].select_dtypes(include=[np.number])
                        corr_matrix = corr_df.corr()
                        st.write(corr_matrix)
                    except Exception as e:
                        st.error(f"Error computing correlation matrix: {str(e)}")

        elif analysis_type == "Time Series Analysis":
            with st.expander('Time Series Analysis', expanded=True):
                # Add time series analysis code here
                st.write("Time series analysis functionality is not implemented yet.")

        elif analysis_type == "Logistic Regression":
            with st.expander('Logistic Regression', expanded=True):
                target_column = st.selectbox("Select Target Variable", df.columns)
                feature_columns = st.multiselect("Select Feature Columns", df.columns)

                if st.button("Run Logistic Regression"):
                    try:
                        X = df[feature_columns]
                        y = df[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write("Confusion Matrix:")
                        st.write(confusion_matrix(y_test, y_pred))
                        st.write("Classification Report:")
                        st.write(classification_report(y_test, y_pred))
                    except Exception as e:
                        st.error(f"Error running logistic regression: {str(e)}")

        elif analysis_type == "Random Forest Classification":
            with st.expander('Random Forest Classification', expanded=True):
                # Add Random Forest code here
                st.write("Random Forest functionality is not implemented yet.")

        elif analysis_type == "Clustering (K-Means)":
            with st.expander('Clustering (K-Means)', expanded=True):
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                feature_columns = st.multiselect("Select Feature Columns", df.columns)

                if st.button("Run K-Means Clustering"):
                    try:
                        X = df[feature_columns]
                        kmeans = KMeans(n_clusters=n_clusters)
                        df['Cluster'] = kmeans.fit_predict(X)
                        st.write(df)
                        st.write("Cluster Centers:")
                        st.write(kmeans.cluster_centers_)
                        st.write("Cluster Labels:")
                        st.write(df['Cluster'].value_counts())
                    except Exception as e:
                        st.error(f"Error running K-Means clustering: {str(e)}")

        elif analysis_type == "ROC Curve":
            with st.expander('ROC Curve', expanded=True):
                target_column = st.selectbox("Select Target Variable", df.columns)
                feature_columns = st.multiselect("Select Feature Columns", df.columns)

                if st.button("Generate ROC Curve"):
                    try:
                        X = df[feature_columns]
                        y = df[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X_train, y_train)
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig = plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc='lower right')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating ROC curve: {str(e)}")

        elif analysis_type == "F1 Score":
            with st.expander('F1 Score', expanded=True):
                target_column = st.selectbox("Select Target Variable", df.columns)
                feature_columns = st.multiselect("Select Feature Columns", df.columns)

                if st.button("Calculate F1 Score"):
                    try:
                        X = df[feature_columns]
                        y = df[target_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        model = LogisticRegression(max_iter=1000)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        st.write(f"F1 Score: {f1}")
                    except Exception as e:
                        st.error(f"Error calculating F1 score: {str(e)}")

        elif analysis_type == "Network Graph Analysis":
            with st.expander('Network Graph Analysis', expanded=True):
                source_column = st.selectbox("Select Source Node Column", df.columns)
                target_column = st.selectbox("Select Target Node Column", df.columns)
                weight_column = st.selectbox("Select Weight Column (Optional)", ["None"] + list(df.columns))

                if st.button("Generate Network Graph"):
                    try:
                        G = nx.Graph()
                        if weight_column != "None":
                            for i, row in df.iterrows():
                                G.add_edge(row[source_column], row[target_column], weight=row[weight_column])
                        else:
                            for i, row in df.iterrows():
                                G.add_edge(row[source_column], row[target_column])
                        
                        pos = nx.spring_layout(G)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, edge_color='gray')
                        
                        if weight_column != "None":
                            edge_labels = nx.get_edge_attributes(G, 'weight')
                            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating network graph: {str(e)}")

        else:
            st.error("Please select an analysis type.")

    else:
        st.warning("Please upload a valid file.")



    bottom_text = """
    <div style='width: 100%; color: white; text-align: center; padding: 10px;'>
        <div>© 2024 Telecom EDA Web App. All rights reserved  by Ranjan Singh(Data Scientist)</div>
        <div>
            <a href='https://www.linkedin.com/in/ranjan-singh-8ab3ba1a4/' target='_blank' style='color: #1DA1F2;'>LinkedIn</a> | 
            <a href='https://github.com/ranjandatascientist' target='_blank' style='color: #333;'>GitHub</a>
        </div>
    </div>
    """

    st.markdown(bottom_text, unsafe_allow_html=True)

# Note: Streamlit does not natively support toast messages, but you can use the `st.success`, `st.info`, `st.warning`, and `st.error` functions to display messages to the user.
# For more interactive messages, you may need to use a custom JavaScript solution or an external library.
