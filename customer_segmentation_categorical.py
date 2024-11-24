import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import plotly.express as px

# Title
st.title("Customer Segmentation Tool (with Categorical Features)")

# File Upload Section
st.sidebar.header("Upload your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data)
    
    # Identify numeric and categorical columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    st.sidebar.header("Clustering Options")
    
    # Select features (numeric and categorical)
    selected_features = st.sidebar.multiselect("Select features for clustering", numeric_columns + categorical_columns)
    
    if len(selected_features) > 1:
        # Separate numeric and categorical features
        numeric_features = [col for col in selected_features if col in numeric_columns]
        categorical_features = [col for col in selected_features if col in categorical_columns]
        
        # Process numeric features
        X_numeric = data[numeric_features].dropna()
        
        # Encode categorical features
        X_categorical = pd.DataFrame()
        if categorical_features:
            encoder = LabelEncoder()
            for col in categorical_features:
                X_categorical[col] = encoder.fit_transform(data[col].astype(str))
        
        # Combine numeric and encoded categorical features
        X_combined = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Handle missing values
        X_combined = X_combined.dropna()
        
        # Standardize the combined features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Select Number of Clusters
        n_clusters = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the dataset
        data['Cluster'] = clusters
        st.write("### Clustered Data")
        st.dataframe(data)
        
        # Visualization
        st.write("### Cluster Visualization")
        if len(X_combined.columns) == 2:
            # Scatter plot for two features
            fig = px.scatter(data, x=X_combined.columns[0], y=X_combined.columns[1], color=data['Cluster'].astype(str),
                             title="Customer Segmentation", labels={'color': 'Cluster'})
            st.plotly_chart(fig)
        else:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = clusters
            fig = px.scatter(pca_df, x='PCA1', y='PCA2', color=pca_df['Cluster'].astype(str),
                             title="Customer Segmentation (PCA Reduced)", labels={'color': 'Cluster'})
            st.plotly_chart(fig)
        
        # Download Segmented Data
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Clustered Data", data=csv, file_name="segmented_data.csv", mime="text/csv")
    else:
        st.warning("Please select at least two features for clustering.")
else:
    st.info("Awaiting CSV file to be uploaded. Upload a file to get started!")
