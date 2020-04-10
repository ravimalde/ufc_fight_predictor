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
    df['name'],
    index=1841)

option2 = st.sidebar.selectbox(
    'Fighter 2',
    df['name'],
    index=1020)

st.header(option1 + '    vs    ' + option2)

st.subheader(option1 + "'s Statistics:")
fighter1_stats = df.loc[df['name'] == option1]
fighter1_stats

st.subheader(option2 + "'s Statistics:")
fighter2_stats = df.loc[df['name'] == option2]
fighter2_stats

if st.button('PREDICT'):

    fighter1_stats_copy = fighter1_stats.copy()
    fighter2_stats_copy = fighter1_stats.copy()

    fighter1_stats.reset_index(drop=True, inplace=True)
    fighter2_stats.reset_index(drop=True, inplace=True)

    df_full1 = fighter1_stats.join(fighter2_stats, lsuffix='_x', rsuffix='_y')
    df_full2 = fighter2_stats.join(fighter1_stats, lsuffix='_x', rsuffix='_y')
    
    df_full1['stance_x_Open Stance'] = df_full1['stance_x'].apply(lambda x: 1 if (x == 'Open Stance') else 0)
    df_full1['stance_x_Orthodox'] = df_full1['stance_x'].apply(lambda x: 1 if (x == 'Orthodox') else 0)
    df_full1['stance_x_Sideways'] = df_full1['stance_x'].apply(lambda x: 1 if (x == 'Sideways') else 0)
    df_full1['stance_x_Southpaw'] = df_full1['stance_x'].apply(lambda x: 1 if (x == 'Southpaw') else 0)
    df_full1['stance_x_Switch'] = df_full1['stance_x'].apply(lambda x: 1 if (x == 'Switch') else 0)

    df_full1['stance_y_Open Stance'] = df_full1['stance_y'].apply(lambda x: 1 if (x == 'Open Stance') else 0)
    df_full1['stance_y_Orthodox'] = df_full1['stance_y'].apply(lambda x: 1 if (x == 'Orthodox') else 0)
    df_full1['stance_y_Sideways'] = df_full1['stance_y'].apply(lambda x: 1 if (x == 'Sideways') else 0)
    df_full1['stance_y_Southpaw'] = df_full1['stance_y'].apply(lambda x: 1 if (x == 'Southpaw') else 0)
    df_full1['stance_y_Switch'] = df_full1['stance_y'].apply(lambda x: 1 if (x == 'Switch') else 0)

    df_full2['stance_y_Open Stance'] = df_full2['stance_y'].apply(lambda x: 1 if (x == 'Open Stance') else 0)
    df_full2['stance_y_Orthodox'] = df_full2['stance_y'].apply(lambda x: 1 if (x == 'Orthodox') else 0)
    df_full2['stance_y_Sideways'] = df_full2['stance_y'].apply(lambda x: 1 if (x == 'Sideways') else 0)
    df_full2['stance_y_Southpaw'] = df_full2['stance_y'].apply(lambda x: 1 if (x == 'Southpaw') else 0)
    df_full2['stance_y_Switch'] = df_full2['stance_y'].apply(lambda x: 1 if (x == 'Switch') else 0)

    df_full2['stance_x_Open Stance'] = df_full2['stance_x'].apply(lambda x: 1 if (x == 'Open Stance') else 0)
    df_full2['stance_x_Orthodox'] = df_full2['stance_x'].apply(lambda x: 1 if (x == 'Orthodox') else 0)
    df_full2['stance_x_Sideways'] = df_full2['stance_x'].apply(lambda x: 1 if (x == 'Sideways') else 0)
    df_full2['stance_x_Southpaw'] = df_full2['stance_x'].apply(lambda x: 1 if (x == 'Southpaw') else 0)
    df_full2['stance_x_Switch'] = df_full2['stance_x'].apply(lambda x: 1 if (x == 'Switch') else 0)

    df_full1.drop(columns=['name_x','stance_x','dob_x','wins_x','losses_x','draws_x',
                          'name_y','stance_y','dob_y','wins_y','losses_y','draws_y'], inplace=True)
    df_full2.drop(columns=['name_x','stance_x','dob_x','wins_x','losses_x','draws_x',
                          'name_y','stance_y','dob_y','wins_y','losses_y','draws_y'], inplace=True)

    x_num_cols = [col for col in df_full1.columns if '_x' in col and 'stance'not in col]
    y_num_cols = [col for col in df_full1.columns if '_y' in col and 'stance'not in col]

    scaler = load(open('scaler.pkl', 'rb'))
    df_full1[x_num_cols] = scaler.transform(df_full1[x_num_cols])
    df_full1[y_num_cols] = scaler.transform(df_full1[y_num_cols])
    df_full2[x_num_cols] = scaler.transform(df_full2[x_num_cols])
    df_full2[y_num_cols] = scaler.transform(df_full2[y_num_cols])

    model = load(open('model.pkl', 'rb'))
    prediction1 = model.predict(df_full1)[0]
    prediction2 = model.predict(df_full2)[0]

    prob1 = model.predict_proba(df_full1)
    prob1_1 = round(prob1[0][0], 2)
    prob2_1 = round(prob1[0][1], 2)

    prob2 = model.predict_proba(df_full2)
    prob1_2 = round(prob2[0][0], 2)
    prob2_2 = round(prob2[0][1], 2)

    if str(option1) == str(option2):
        st.subheader("The same fighter is selected. Please select two different fighters.")
    elif prob1_1 > prob2_1 and prob1_1 > prob1_2 and prob1_1 > prob2_2:
        st.subheader(str(option2) + " wins with a probability of " + str(prob1_1))
    elif prob2_1 > prob1_1 and prob2_1 > prob1_2 and prob2_1 > prob2_2:
        st.subheader(str(option1) + " wins with a probability of " + str(prob2_1))
    elif prob1_2 > prob1_1 and prob1_2 > prob2_1 and prob1_2 > prob2_2:
        st.subheader(str(option1) + " wins with a probability of " + str(prob1_2))
    elif prob2_2 > prob1_1 and prob2_2 > prob1_2 and prob2_2 > prob2_1:
        st.subheader(str(option2) + " wins with a probability of " + str(prob2_2))
else:
    pass

fighter1_wins = df['wins'].loc[df['name'] == option1].values[0]
fighter1_losses = df['losses'].loc[df['name'] == option1].values[0]
fighter1_draws = df['draws'].loc[df['name'] == option1].values[0]

fighter2_wins = df['wins'].loc[df['name'] == option2].values[0]
fighter2_losses = df['losses'].loc[df['name'] == option2].values[0]
fighter2_draws = df['draws'].loc[df['name'] == option2].values[0]

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

stat_descriptions = "str_landed_per_min: Strikes landed per minute over the last 5 fights.\n\
str_acc: Percentage of attempted strikes that landed over the last 5 fights.\n\
str_absorb_per_min: Strikes from opponent that landed per minute over the last 5 fights.\n\
str_def: Percentage of opponent strikes that did not land over the last 5 fights.\n\
td_avg: Average takedowns landed per 15 mins over the last 5 fights.\n\
td_acc: Percentage of attempted takedowns that landed over the last 5 fights.\n\
td_def: Percentage of opponent takedowns that did not land over the last 5 fights.\n\
sub_avg: Average submissions attempted per 15 mins over the last 5 fights."

if option3 == True:
    st.text(stat_descriptions)
else:
    pass


