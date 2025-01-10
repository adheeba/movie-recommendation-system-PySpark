import streamlit as st
import pandas as pd
from PIL import Image
from pyspark.sql.functions import explode

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#st.set_page_config(layout="wide")
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model  
spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()
model = ALSModel.load("./model")
# Load data
movies_all = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("./data/movies.csv")
image, title = st.columns([0.8,2])
with image:
  image = Image.open('./streamlit/property.png')
  st.image(image, width=200)
with title:
  st.title(':rainbow[Movie Recommendation System]')

  st.subheader("Get our movie recommendations to enjoy next shows")
col1, col2, col3 = st.columns(3)


@st.cache_data
def getMovieRecommendationForUser():  
    try:
        #prediction = model.transform()
        user_recs = model.recommendForAllUsers(50)
        recommendationsDF = (user_recs
            .select("userId", explode("recommendations")
            .alias("recommendation"))
            .select("userId", "recommendation.*")
          )
        recommendationsDF.show(5)
        #print(type(recommendationsDF))
        #prediction.show(5)#.toPandas()
        return recommendationsDF.toPandas()
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None
result = getMovieRecommendationForUser()
with col2:
  userId = st.selectbox(':blue[Enter your user Id?]', options=result['userId'].unique())
c0l1,col2,col3 = st.columns([10,2,10])
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
      result = getMovieRecommendationForUser()
      recommendation = result.query(f'userId=={userId}')
      
      
      movies = pd.merge(recommendation, movies_all.toPandas(), on='movieId', how='inner')
      if result is not None:
        with col21:
          st.balloons()
          st.success(f"### Movie Recommendation for the users are")
          st.dataframe(movies)

