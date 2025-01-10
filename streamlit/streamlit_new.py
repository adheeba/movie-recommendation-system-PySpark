import streamlit as st
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Streamlit page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Spark session
spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()

# Load the pre-trained ALS model
try:
    model = ALSModel.load("D:\\Becode\\Projects\\movie-recommendation-system-PySpark\\model")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header section
image, title = st.columns([0.8, 2])
with image:
    try:
        logo = Image.open('./streamlit/property.png')
        st.image(logo, width=200)
    except FileNotFoundError:
        st.warning("Logo image not found.")

with title:
    st.title(':rainbow[Movie Recommendation System]')
    st.subheader("Get our movie recommendations to enjoy next shows")

# User ID input
col1, col2, col3 = st.columns(3)
with col2:
    userId = st.number_input(':blue[Enter your user Id?]', min_value=1, step=1)

# Recommendation function
@st.cache_data
def getMovieRecommendationForUser(_data):
    try:
        prediction = model.transform(_data)
        return prediction.toPandas()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None

# Recommendation button
if st.button('Recommend Movies'):
    if userId > 0:
        input_data = spark.createDataFrame(
            [(int(userId), 10, 4.4)], 
            ["userId", "movieId", "rating"]
        )
        result = getMovieRecommendationForUser(input_data)
        if result is not None:
            st.success("### Movie Recommendation for the user:")
            #st.dataframe(result)
            st.balloons()
    else:
        st.warning("Please enter a valid User ID.")
