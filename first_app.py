import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from pickle import load

st.title("UFC Fight Predictor")

df = pd.read_csv("data_for_database.csv")

option1 = st.sidebar.selectbox(
    'Fighter 1',
     df['name'])

option2 = st.sidebar.selectbox(
    'Fighter 2',
     df['name'])

'Fighters Selected:', option1,' vs ', option2

fighter1_wins = str(df['wins'].loc[df['name'] == option1].values[0])
fighter1_losses = str(df['draws'].loc[df['name'] == option1].values[0])
fighter1_draws = str(df['losses'].loc[df['name'] == option1].values[0])

fighter2_wins = str(df['wins'].loc[df['name'] == option2].values[0])
fighter2_losses = str(df['draws'].loc[df['name'] == option2].values[0])
fighter2_draws = str(df['losses'].loc[df['name'] == option2].values[0])

st.text(option1 + ' MMA record: ' + fighter1_wins + '-' + fighter1_losses + '-' + fighter1_draws)
st.text(option2 + ' MMA record: ' + fighter2_wins + '-' + fighter2_losses + '-' + fighter2_draws)

option4 = st.sidebar.checkbox(
    'Static Fighter Stat Plots',
    value=True
)

option5 = st.sidebar.checkbox(
    'Dynamic Fighter Stat Plots',
)

f1_height = df['height'][df['name'] == option1].values[0]
f2_height = df['height'][df['name'] == option2].values[0]

f1_reach = df['reach'][df['name'] == option1].values[0]
f2_reach = df['reach'][df['name'] == option2].values[0]

if option4 == True:

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[1], str(option1).split(' ')[1]], [f2_height, f1_height], color=('red','blue'))
    plt.xlim(1)
    plt.title("Fighter Heights (m)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[1], str(option1).split(' ')[1]], [f2_reach, f1_reach], color=('red','blue'))
    plt.xlim(1)
    plt.title("Fighter Reach (m)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

else:
    pass

option3 = st.sidebar.checkbox(
    'Dynamic Fighter Stat Definitions',
)

stat_descriptions = "str_landed_per_min: Average strikes landed per minute in the last 5 fights.\nstr_def: Percentage strikes from opponent that did not land in the last 5 fights.\nstr_absorb_per_min: Average strikes from opponent that landed per minute over the last 5 fights."

if option3 == True:
    st.text(stat_descriptions)
else:
    pass

if st.button('PREDICT'):

    fighter1_stats = df.loc[df['name'] == option1]
    fighter2_stats = df.loc[df['name'] == option2]

    new_cols = ['height_diff','reach_diff','weight_diff','age_diff','str_landed_per_min_diff','str_acc_diff',
                'str_absorb_per_min_diff','str_def_diff','td_avg_diff','td_acc_diff','td_def_diff',
                'sub_avg_diff','win_percentage_diff']

    old_cols = ['height','reach','weight','age','str_landed_per_min','str_acc',
                'str_absorb_per_min','str_def','td_avg','td_acc','td_def',
                'sub_avg','win_percentage']
    
    df_diff = pd.DataFrame()

    for i in range(len(new_cols)):
        df_diff[new_cols[i]] = fighter1_stats[old_cols[i]].values - fighter2_stats[old_cols[i]].values

    scaler = load(open('scaler.pkl', 'rb'))
    scaled_data = scaler.transform(df_diff)

    model = load(open('model.pkl', 'rb'))
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.text("Fighter 1 wins!")
    else:
        st.text("Fighter 2 wins!")

else:
    pass




