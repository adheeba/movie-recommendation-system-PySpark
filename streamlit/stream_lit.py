import streamlit as st
import pandas as pd
from PIL import Image
from pyspark.sql.functions import explode

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import os
import sys


#-------------------streamlit code for page setup---------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

#----------pyspark initial code for session creation and loading data -------
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()
# Load saved ALS model
try:
    model = ALSModel.load("./model")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# Load movies data to show the names of movie based on the movie Id recieved
movies_all = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("./data/movies.csv")

@st.cache_data
def getMovieRecommendationForUser():  
    try:
        user_recs = model.recommendForAllUsers(50)
        recommendationsDF = (user_recs
            .select("userId", explode("recommendations")
            .alias("recommendation"))
            .select("userId", "recommendation.*")
          )
        return recommendationsDF.toPandas()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None

result = getMovieRecommendationForUser()

#   ----- streamlit code here ------
image, title = st.columns([0.8,2])
with title:
    st.title(':rainbow[Movie Recommendation System]')
    st.subheader("Get our movie recommendations to enjoy next shows")

    try:
        image = Image.open('./streamlit/movie.jpg')
        st.image(image, width=800)
    except FileNotFoundError:
        st.warning("Logo image not found.")
col1, col2, col3 = st.columns(3)
with col2:
  userId = st.selectbox(':blue[Enter your user Id?]', options=result['userId'].unique())
col1,col2,col3 = st.columns([10,2,10])
col11,col21,col31 = st.columns([0.5,1,0.3])

with col2:
  st.markdown("""
<style>.element-container:has(#button-after) + div button {
 background-color: blue;
 p{
 color : white;
  }
 }</style>""", unsafe_allow_html=True)
  st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
  if st.button('Recommend Movies'):
      if result is not None:
        recommendation = result.query(f'userId=={userId}')
        movies = pd.merge(recommendation, movies_all.toPandas(), on='movieId', how='inner')
        with col21:
            st.balloons()
            st.success(f"### Movie Recommendation for the user {userId} are")
            st.markdown("""
                <style>
                table {background-color: blue;}
                </style>
                """, unsafe_allow_html=True)
            st.dataframe(movies.style.background_gradient(cmap='Blues', axis=0))
      else:
          st.error("error in prediction")