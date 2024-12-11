import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(
    page_title="MovieLens Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load data
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        'ml-1m/ratings.dat',
        sep='::',
        header=None,
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
    )
    movies = pd.read_csv(
        'ml-1m/movies.dat',
        sep='::',
        header=None,
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='ISO-8859-1',
    )
    users = pd.read_csv(
        'ml-1m/users.dat',
        sep='::',
        header=None,
        engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
    )
    return ratings, movies, users


# Load data
ratings, movies, users = load_data()

# Merge datasets
data = pd.merge(pd.merge(ratings, users), movies)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    **Navigate to different sections:**
    - 🏠 **Overview**: Dataset structure and introduction.
    - 🔍 **Data Exploration**: Explore and filter data interactively.
    - 📈 **Clustering Analysis**: Perform clustering and visualize results.
    - 📝 **Conclusions**: Insights and recommendations.
    """
)
options = st.sidebar.radio("Choose Section", ["🏠 Overview", "🔍 Data Exploration", "📈 Clustering Analysis", "📝 Conclusions"])

# Optimize spacing in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Project:** MovieLens Data Analysis")
st.sidebar.markdown("👨‍💻 **Developed by:** Sheeshable Guys")
st.sidebar.markdown("""
**Team Members:**
- Españo, Trisan Jae
- Tompar, Simoun Cloyd
- Villas, Steve Laurenz
- Sanico, Kade
- Fernandez, Chris Marklen
""")

# Overview Section
if options == "🏠 Overview":
    st.title("🎥 MovieLens Data Analysis")
    st.write(
 """
    ### Welcome to the MovieLens Data Analysis App!
    Explore and analyze the **MovieLens 1M dataset** interactively, with a focus on clustering users based on their movie ratings using the K-means algorithm. Use the sidebar to navigate through the sections.

    #### Dataset Overview:
    The **MovieLens 1M dataset** contains 1 million movie ratings provided by users. It is composed of three primary components:
    - **Ratings**: User ratings for various movies.
    - **Movies**: Metadata about movies, including titles and genres.
    - **Users**: Demographic information about the users, such as gender, age, and occupation.

    #### Research Question:
    How can users be segmented into distinct groups based on their movie rating patterns to understand viewing preferences and inform personalized recommendations?

    #### Selected Analysis Technique:
    This study employs the **K-means clustering algorithm**, a machine learning approach for unsupervised learning, to group users into clusters. The key steps include:
    - Creating a user-item matrix from the dataset.
    - Determining the optimal number of clusters using the **Elbow Method**.
    - Applying K-means clustering to segment users.
    - Using **PCA (Principal Component Analysis)** to visualize the clustering results.

    #### Benefits:
    This analysis helps uncover hidden patterns in user behavior, which can be leveraged to:
    - Provide personalized movie recommendations.
    - Design targeted marketing campaigns for specific user segments.
    """
    )

    # Dataset structure in columns to save space
    st.subheader("📂 Dataset Structure")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Ratings Dataset**")
        st.dataframe(ratings.head(5), height=250 )
    with col2:
        st.markdown("**Movies Dataset**")
        st.dataframe(movies.head(5), height=250)
    with col3:
        st.markdown("**Users Dataset**")
        st.dataframe(users.head(5), height=250)

# Data Exploration Section
elif options == "🔍 Data Exploration":
    st.title("🔍 Data Exploration and Preparation")

    # Filters in a compact two-column layout
    st.markdown("### 🎯 Filter Data by Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.selectbox("Select Gender:", ["All", "M", "F"])
    with col2:
        age_filter = st.slider("Select Age Range:", 1, 50, (1, 50))

    filtered_data = data
    if gender_filter != "All":
        filtered_data = filtered_data[filtered_data["Gender"] == gender_filter]
    filtered_data = filtered_data[
        (filtered_data["Age"] >= age_filter[0]) & (filtered_data["Age"] <= age_filter[1])
    ]

    st.markdown(f"Displaying **{gender_filter}** users in the age range **{age_filter[0]} to {age_filter[1]}**:")
    st.dataframe(filtered_data.head(10), height=250)

    # Display descriptive statistics and data distribution side by side
    st.markdown("### 📊 Descriptive Statistics and Data Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(filtered_data.describe(), height=250)
    with col2:
        st.markdown("**Data Distribution**")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(filtered_data["Gender"], ax=ax[0])
        ax[0].set_title("Gender Distribution")
        sns.histplot(filtered_data["Age"], bins=10, kde=True, ax=ax[1])
        ax[1].set_title("Age Distribution")
        st.pyplot(fig)

    # Compact movie search section
    st.markdown("### 🔍 Search for a Movie")
    movie_search = st.text_input("Enter movie title:")
    if movie_search:
        filtered_movies = movies[
            movies["Title"].str.contains(movie_search, case=False, na=False)
        ]
        st.dataframe(filtered_movies[["Title", "Genres"]])
        if not filtered_movies.empty:
            movie_id = filtered_movies.iloc[0]["MovieID"]
            movie_ratings = ratings[ratings["MovieID"] == movie_id]
            avg_rating = movie_ratings["Rating"].mean()
            st.markdown(f"**Average Rating:** {avg_rating:.2f}")

# Clustering Analysis Section
elif options == "📈 Clustering Analysis":
    st.title("📈 Clustering Analysis")




    #   Relevance to Research Question:
    #   This matrix provides the foundation for segmenting users based on their rating patterns, 
    #   answering the "movie rating patterns" part of the question.

    st.markdown("### 🎯 User-Item Matrix Preparation")
    user_item_matrix = data.pivot_table(index="UserID", columns="MovieID", values="Rating").fillna(0)

    # Optimal number of clusters with adjusted layout

    # Relevance to Research Question:
    # This ensures the clustering process is not biased by scale differences in movie ratings.

    st.markdown("### ⚙️ Determine Optimal Number of Clusters")
    scaler = StandardScaler()
    user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

    # Relevance to Research Question:
    # This step ensures the segmentation is meaningful and avoids overfitting or underfitting the clusters.

    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_item_matrix_scaled)
        inertia.append(kmeans.inertia_)



    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K, inertia, "bo-")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method For Optimal k")
    st.pyplot(fig)


    # Clustering details and profiling

    # Relevance to Research Question:
    # This directly segments users into distinct groups, addressing the "segmentation" aspect of the question.
    st.markdown("### ✨ Clustering Details and Profiling")
    optimal_k = st.slider("Select the number of clusters:", 2, 10, 5)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(user_item_matrix_scaled)
    user_item_matrix["Cluster"] = clusters
    st.dataframe(user_item_matrix.head(), height=250)

    # Cluster profiling and visualization side by side
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Cluster Profiling**")
        cluster_profile = user_item_matrix.groupby("Cluster").mean()
        st.dataframe(cluster_profile, height=250)
    with col2:
        st.markdown("**Cluster Visualization (PCA)**")

    # Relevance to Research Question:
    # This visually validates the distinctness of clusters, providing evidence of meaningful segmentation.
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(user_item_matrix_scaled)
        principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
        principal_df["Cluster"] = clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=principal_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax)
        ax.set_title("Clusters Visualization")
        st.pyplot(fig)

# Conclusions Section
elif options == "📝 Conclusions":
    st.title("📝 Conclusions and Recommendations")
    st.markdown(
        """
        **Key Takeaways:**
        - Users are grouped into clusters based on their movie rating patterns.
        - Distinct clusters reveal preferences for different genres or types of movies.
        - Insights from clustering can be used to build personalized movie recommendation systems.

        **Recommendations:**
        - Develop targeted campaigns for user groups with similar preferences.
        - Enhance user engagement by recommending movies tailored to user preferences.
        """
    )
