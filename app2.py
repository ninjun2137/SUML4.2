import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

sex_d = {0:"Kobieta", 1:"Mężczyzna"}
chestpaintype_d = {0:"ASY",1:"ATA",2:"NAP",3:"TA"}
restingECG_d = {0:"LVH",1:"Normal",2:"ST"}
exerciseAngina_d = {0:"Nie",1:"Tak"}
stSlope_d = {0:"Spada",1:"Płaski",2:"Rośnie"}
fastingbs_d = {0:"Nie",1:"Tak"}
 
def main():
    st.set_page_config(page_title="Czy masz epicką chorobę serca?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://cdn.aptelia.pl/image/a/article/57c588cd-scalecrop-840x350.jpg")

    with overview:
        st.title("Czy masz epicką chorobę serca?")

    with left:
        chestpaintype_radio = st.radio("Typ bólu:", list(chestpaintype_d.keys()), format_func=lambda x : chestpaintype_d[x] )
        sex_radio = st.radio( "Płeć:", list(sex_d.keys()), format_func= lambda x: sex_d[x] )
        exerciseAngina_radio = st.radio( "Angina ćwiczeniowa:", list(exerciseAngina_d.keys()), format_func= lambda x: exerciseAngina_d[x] )
        restingECG_radio = st.radio("Resting ecg:", list(restingECG_d.keys()), format_func= lambda x: restingECG_d[x])
        stSlope_radio = st.radio("St slope:", list(stSlope_d.keys()), format_func= lambda x: stSlope_d[x])
        fastingbs_radio = st.radio("Fasting BS:", list(fastingbs_d.keys()), format_func= lambda x: fastingbs_d[x])
        
    with right:
        age_slider = st.slider("Wiek", value=28, min_value=28, max_value=80)
        restingBP_slider = st.slider("Ciśnienie krwi w spoczynku:",min_value=80,max_value=200)
        cholesterik_slider = st.slider("Cholesterol:",min_value=80,max_value=700)
        maxHr_slider = st.slider("MaxHR:",min_value=60,max_value=205)
        oldpeak_slider = st.slider("Oldpeak:",min_value=-4.0,max_value=7.0,step=0.1)

    data = [[age_slider,sex_radio,chestpaintype_radio,restingBP_slider,cholesterik_slider,fastingbs_radio,restingECG_radio,maxHr_slider,exerciseAngina_radio,oldpeak_slider,stSlope_radio]]
    choroba = model.predict(data)
    c_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba ma chorobę serca?")
        st.subheader(("Tak" if choroba[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(c_confidence[0][choroba][0] * 100))

if __name__ == "__main__":
    main()
