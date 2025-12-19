import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import json

DATA = "welcome_survey_simple_v1.csv"
MODEL_NAME = "welcome_survey_clustering_pipeline_v1"
CLUSTER_NAMES_AND_DESCRIPTIONS = "welcome_survey_cluster_names_and_descriptions_v1.json"
 
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)
   
    return df_with_clusters

@st.cache_data
def get_cluster_name_and_descriptions():
    with open (CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
               return json.load(f)
               


st.title("Znajdź znajomych")

# ### ➡ To zastępuje st.title() i daje pełną kontrolę nad wyrównaniem:
# st.markdown(
#     "<h1 style='text-align: right;'>Znajdź znajomych v.2  </h1>",
#     unsafe_allow_html=True 
# )
# ###

with st.sidebar:
    st.header("Powiedz coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby o podobnych zainteresowaniach")
    age = st.selectbox("Wiek", ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65"])
    edu_level = st.selectbox("Wykształcenie", ["Podstawowe", "Średnie", "Wyższe"])
    fav_animal = st.selectbox("Ulubiony zwierzak", ["Brak ulubionych", "Psy", "Koty", "Inne", "Psy i Koty"])
    fav_place = st.selectbox("Ulubione miejsce", ["Nad wodą", "W lesie", "W górach", "inne"])
    gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])

person_df = pd.DataFrame([
    {
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animal,
        'fav_place': fav_place,
        'gender': gender
    }
])

model = get_model()
all_df = get_all_participants()
cluster_name_and_descriptions = get_cluster_name_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)['Cluster'][0]
predicted_cluster_data = cluster_name_and_descriptions[predicted_cluster_id]


st.header(f"Najbliżej Ci do klastra: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
st.metric("Liczba osób w tym klastrze:", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(
    same_cluster_df.sort_values("age"), x="age")
fig .update_layout(
    title="Rozkład wieku w grupie podobnych zainteresowań",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig) 

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Poziom wykształcenia",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Ulubione zwierzaki w grupie",
    xaxis_title="Ulubione zwierzaki",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Ulubione miejsca w grupie",
    xaxis_title="Ulubione miejsca",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Płeć w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)