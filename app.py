import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).parent

DATA = BASE_DIR / "welcome_survey_simple_v1.csv"
MODEL_FILE = BASE_DIR / "cluster_pipe.joblib"
CLUSTER_NAMES_AND_DESCRIPTIONS = BASE_DIR / "welcome_survey_cluster_names_and_descriptions_v1.json"

st.set_page_config(page_title="Znajdź znajomych", layout="centered")


@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILE)


def safe_predict(model, df: pd.DataFrame):
    """
    Bezpieczna predykcja dla modelu zapisanego w joblib.
    Wariant docelowy: sklearn pipeline z metodą .predict().
    """
    if hasattr(model, "predict"):
        return model.predict(df)

    raise RuntimeError(
        "Model w cluster_pipe.joblib nie ma metody predict(). "
        "Upewnij się, że zapisałeś sklearn pipeline (a nie wrapper wymagający PyCaret)."
    )


@st.cache_data
def get_cluster_name_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_all_participants_with_clusters():
    df = pd.read_csv(DATA, sep=";")
    model = get_model()
    df = df.copy()
    df["Cluster"] = safe_predict(model, df)
    return df


st.title("Znajdź znajomych")

with st.sidebar:
    st.header("Powiedz coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby o podobnych zainteresowaniach")

    age = st.selectbox("Wiek", ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65"])
    edu_level = st.selectbox("Wykształcenie", ["Podstawowe", "Średnie", "Wyższe"])
    fav_animal = st.selectbox("Ulubiony zwierzak", ["Brak ulubionych", "Psy", "Koty", "Inne", "Psy i Koty"])
    fav_place = st.selectbox("Ulubione miejsce", ["Nad wodą", "W lesie", "W górach", "inne"])
    gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])

# Kolumny zgodne z Twoim CSV:
person_df = pd.DataFrame([{
    "age": age,
    "edu_level": edu_level,
    "fav_animals": fav_animal,
    "fav_place": fav_place,
    "gender": gender,
}])

st.write("Twoje dane:")
st.dataframe(person_df, hide_index=True)

model = get_model()
all_df = get_all_participants_with_clusters()
cluster_info = get_cluster_name_and_descriptions()

predicted_cluster_id = int(safe_predict(model, person_df)[0])

# JSON często ma klucze jako stringi "0","1",...
predicted_cluster_data = cluster_info.get(
    str(predicted_cluster_id),
    {"name": f"Klaster {predicted_cluster_id}", "description": "Brak opisu klastra."}
)

st.header(f"Najbliżej Ci do klastra: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data["description"])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba osób w tym klastrze:", int(len(same_cluster_df)))

st.header("Osoby z grupy")

fig = px.histogram(same_cluster_df, x="age")
fig.update_layout(title="Rozkład wieku w grupie", xaxis_title="Wiek", yaxis_title="Liczba osób")
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Poziom wykształcenia",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Ulubione zwierzaki w grupie",
    xaxis_title="Ulubione zwierzaki",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Ulubione miejsca w grupie",
    xaxis_title="Ulubione miejsca",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(title="Płeć w grupie", xaxis_title="Płeć", yaxis_title="Liczba osób")
st.plotly_chart(fig, use_container_width=True)
