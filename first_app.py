import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load
import sklearn

st.title("UFC Fight Predictor")

df = pd.read_csv("data_for_database.csv")

option1 = st.sidebar.selectbox(
    'Fighter 1',
    df['name'])

option2 = st.sidebar.selectbox(
    'Fighter 2',
    df['name'])

st.header(option1 + '    vs    ' + option2)

st.subheader(option1 + "'s Statistics:")
fighter1_stats = df.loc[df['name'] == option1]
fighter1_stats

st.subheader(option2 + "'s Statistics:")
fighter2_stats = df.loc[df['name'] == option2]
fighter2_stats

if st.button('PREDICT'):

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

    prob = model.predict_proba(scaled_data)
    prob1 = round(prob[0][1], 2)
    prob2 = round(1-prob1, 2)

    if prediction == 1:
        st.subheader(str(option1) + " wins with a probability of " + str(prob1))
    elif prediction == 0:
        st.subheader(str(option2) + " wins with a probability of " + str(prob2))
else:
    pass


fighter1_wins = df['wins'].loc[df['name'] == option1].values[0]
fighter1_losses = df['draws'].loc[df['name'] == option1].values[0]
fighter1_draws = df['losses'].loc[df['name'] == option1].values[0]

fighter2_wins = df['wins'].loc[df['name'] == option2].values[0]
fighter2_losses = df['draws'].loc[df['name'] == option2].values[0]
fighter2_draws = df['losses'].loc[df['name'] == option2].values[0]

labels = ['Draws','Losses','Wins']
fighter1_record = [fighter1_draws, fighter1_losses, fighter1_wins]
fighter2_record = [fighter2_draws, fighter2_losses, fighter2_wins]

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=SMALL_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)

ind = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(16,6))
ax.barh(ind, fighter1_record, width, label=str(option1), color='royalblue')
ax.barh(ind + width, fighter2_record, width, label=str(option2), color='tomato')
ax.set(yticks=ind+width/2, yticklabels=labels, ylim=[2*width - 1, len(labels)])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=2)
st.pyplot()

option4 = st.sidebar.checkbox(
    'Static Statistic Plots',
    value=True
)

option5 = st.sidebar.checkbox(
    'Standing Fighter Stat Plots',
)

option6 = st.sidebar.checkbox(
    'Ground Fighter Stat Plots'
)

f1_height = df['height'][df['name'] == option1].values[0]
f2_height = df['height'][df['name'] == option2].values[0]

f1_reach = df['reach'][df['name'] == option1].values[0]
f2_reach = df['reach'][df['name'] == option2].values[0]

f1_age = df['age'][df['name'] == option1].values[0]
f2_age = df['age'][df['name'] == option2].values[0]

if option4 == True:

    st.subheader("Static Fighter Statistics:")

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_height, f1_height], color=('tomato','royalblue'))
    plt.xlim(1.2)
    plt.title("Fighter Height (m)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_reach, f1_reach], color=('tomato','royalblue'))
    plt.xlim(1.2)
    plt.title("Fighter Reach (m)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_age, f1_age], color=('tomato','royalblue'))
    plt.xlim(18)
    plt.title("Fighter Age (years)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

else:
    pass

f1_str_landed_per_min = df['str_landed_per_min'][df['name'] == option1].values[0]
f2_str_landed_per_min = df['str_landed_per_min'][df['name'] == option2].values[0]

f1_str_acc = df['str_acc'][df['name'] == option1].values[0]
f2_str_acc = df['str_acc'][df['name'] == option2].values[0]

f1_str_absorbed_per_min = df['str_absorb_per_min'][df['name'] == option1].values[0]
f2_str_absorbed_per_min = df['str_absorb_per_min'][df['name'] == option2].values[0]

f1_str_def = df['str_def'][df['name'] == option1].values[0]
f2_str_def = df['str_def'][df['name'] == option2].values[0]

if option5 == True:

    st.subheader("Standing Fighter Statistics:")

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_str_landed_per_min, f1_str_landed_per_min], color=('tomato','royalblue'))
    plt.title("Strikes Landed per Minute", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_str_acc, f1_str_acc], color=('tomato','royalblue'))
    plt.title("Strike Accuracy (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_str_absorbed_per_min, f1_str_absorbed_per_min], color=('tomato','royalblue'))
    plt.title("Strikes Absorbed per Minute", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_str_def, f1_str_def], color=('tomato','royalblue'))
    plt.title("Strike Defence (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

else:
    pass

f1_td_avg = df['td_avg'][df['name'] == option1].values[0]
f2_td_avg = df['td_avg'][df['name'] == option2].values[0]

f1_td_acc = df['td_acc'][df['name'] == option1].values[0]
f2_td_acc = df['td_acc'][df['name'] == option2].values[0]

f1_td_def = df['td_def'][df['name'] == option1].values[0]
f2_td_def = df['td_def'][df['name'] == option2].values[0]

f1_sub_avg = df['sub_avg'][df['name'] == option1].values[0]
f2_sub_avg = df['sub_avg'][df['name'] == option2].values[0]

if option6 == True:

    st.subheader("Ground Fighter Statistics:")

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_td_avg, f1_td_avg], color=('tomato','royalblue'))
    plt.title("Takedowns per 15 mins", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_td_acc, f1_td_acc], color=('tomato','royalblue'))
    plt.title("Takedown Accuracy (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_td_def, f1_td_def], color=('tomato','royalblue'))
    plt.title("Takedown Defence (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

    plt.figure(figsize=(16,3))
    plt.barh([str(option2).split(' ')[-1], str(option1).split(' ')[-1]], [f2_sub_avg, f1_sub_avg], color=('tomato','royalblue'))
    plt.title("Takedown Defence (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot()

else:
    pass

option3 = st.sidebar.checkbox(
    'Show Statistic Definitions',
)

stat_descriptions = "str_landed_per_min: Strikes landed per minute in the last 5 fights.\nstr_def: Percentage strikes from opponent that did not land in the last 5 fights.\nstr_absorb_per_min: Strikes from opponent that landed per minute over the last 5 fights."

if option3 == True:
    st.text(stat_descriptions)
else:
    pass


