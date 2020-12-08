# Verification of Results from Network Capacity Predictor
import numpy
import pandas as pd
import numpy as np
import math
# convert an array of values into a dataset matrix


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

THRESHOLD_USERS = 60
THRESHOLD_PRB = 60

df_resullts = pd.read_csv('results/nw-pred-results.csv')

#'Predicted_Users', 'Predicted_ Downlink_PRB', 'GT_Users', 'GT_Downlink_PRB'


df_resullts['Predicted_Is_Congested_AND'] = np.where((df_resullts['Predicted_Users'] >= THRESHOLD_USERS) & (df_resullts['Predicted_ Downlink_PRB'] >= THRESHOLD_PRB), 1, 0)
df_resullts['GT_Is_Congested_AND'] = np.where((df_resullts['GT_Users'] >= THRESHOLD_USERS) & (df_resullts['GT_Downlink_PRB'] >= THRESHOLD_PRB), 1, 0)
df_resullts['MATCH_AND']  =  np.where((df_resullts['Predicted_Is_Congested_AND'] == df_resullts['GT_Is_Congested_AND']), 1, 0)

df_resullts['AND_MATCH_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_AND'] == 1) & (df_resullts['GT_Is_Congested_AND'] == 1), 1, 0)
df_resullts['AND_MATCH_NO_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_AND'] == 0) & (df_resullts['GT_Is_Congested_AND'] == 0), 1, 0)


df_resullts['AND_MISMATCH_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_AND'] == 1) & (df_resullts['GT_Is_Congested_AND'] == 0), 1, 0)
df_resullts['AND_MISMATCH_NO_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_AND'] == 0) & (df_resullts['GT_Is_Congested_AND'] == 1), 1, 0)


print(df_resullts['Predicted_Is_Congested_AND'].value_counts())
print(df_resullts['GT_Is_Congested_AND'].value_counts())
print(df_resullts['MATCH_AND'].value_counts())

print(df_resullts['AND_MATCH_CONGESTION'].value_counts())
print(df_resullts['AND_MATCH_NO_CONGESTION'].value_counts())

print(df_resullts['AND_MISMATCH_CONGESTION'].value_counts())
print(df_resullts['AND_MISMATCH_NO_CONGESTION'].value_counts())


df_resullts['Predicted_Is_Congested_OR'] = np.where((df_resullts['Predicted_Users'] <= THRESHOLD_USERS) | (df_resullts['Predicted_ Downlink_PRB'] <= THRESHOLD_PRB), 0, 1)
df_resullts['GT_Is_Congested_OR'] = np.where((df_resullts['GT_Users'] <= THRESHOLD_USERS) | (df_resullts['GT_Downlink_PRB'] <= THRESHOLD_PRB), 0, 1)
df_resullts['MATCH_OR']  =  np.where((df_resullts['Predicted_Is_Congested_OR'] == df_resullts['GT_Is_Congested_OR']), 1, 0)


df_resullts['OR_MATCH_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_OR'] == 1) & (df_resullts['GT_Is_Congested_OR'] == 1), 1, 0)
df_resullts['OR_MATCH_NO_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_OR'] == 0) & (df_resullts['GT_Is_Congested_OR'] == 0), 1, 0)


df_resullts['OR_MISMATCH_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_OR'] == 1) & (df_resullts['GT_Is_Congested_OR'] == 0), 1, 0)
df_resullts['OR_MISMATCH_NO_CONGESTION']  =  np.where((df_resullts['Predicted_Is_Congested_OR'] == 0) & (df_resullts['GT_Is_Congested_OR'] == 1), 1, 0)


print(df_resullts['Predicted_Is_Congested_OR'].value_counts())
print(df_resullts['GT_Is_Congested_OR'].value_counts())
print(df_resullts['MATCH_OR'].value_counts())


print(df_resullts['OR_MATCH_CONGESTION'].value_counts())
print(df_resullts['OR_MATCH_NO_CONGESTION'].value_counts())

print(df_resullts['OR_MISMATCH_CONGESTION'].value_counts())
print(df_resullts['OR_MISMATCH_NO_CONGESTION'].value_counts())


#https://futurestud.io/tutorials/matplotlib-stacked-bar-plots


cell = ['Cell - 987765543']
congested_nw =  [float(df_resullts['AND_MATCH_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
uncongested_nw = [float(df_resullts['AND_MATCH_NO_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
congestion_err  = [float(df_resullts['AND_MISMATCH_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
uncongestion_err = [float(df_resullts['AND_MISMATCH_NO_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]

print("AND PEAK MATCH", congested_nw)
print("AND VALLEY MATCH", uncongested_nw)

print("AND PEAK VALLEY MISMATCH", congestion_err)
print("AND VALLEY PEAK MATCH", uncongestion_err)

df2 = pd.DataFrame({'NETWORK_AT_PEAK': congested_nw, 'NETWORK_AT_VALLEY': uncongested_nw, 'PEAK_VALLEY_ERROR': congestion_err, 'VALLEY_PEAK_ERROR': uncongestion_err})
ax = df2.plot(kind='bar', stacked=True, figsize=(8, 6))

annonation_labels = [congested_nw, uncongested_nw, congestion_err, uncongestion_err]
count = 0
# Add this loop to add the annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(str(round(p.get_y() + height + 0.01, 2)) + "%-(" + str(round(annonation_labels[count][0], 2)) + "%)", (p.get_x() + .5 * width, p.get_y() + height + 0.01), ha='center', fontsize=6)
    count = count + 1

plt.xticks(rotation=30)
plt.ylabel("Number Records - Actual vs Predicted Match/Mismatch", fontsize=10, labelpad=12)
plt.xlabel("Congestion Summary", fontsize=10, labelpad=12)
plt.title("Comparing AND Predictions - Actual vs Predicted(%)", fontsize=10, y=1.02)
plt.legend(loc='upper left', frameon=False, prop={'size': 6})
plt.savefig('figs/and.png')

congested_nw =  [float(df_resullts['OR_MATCH_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
uncongested_nw = [float(df_resullts['OR_MATCH_NO_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
congestion_err  = [float(df_resullts['OR_MISMATCH_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]
uncongestion_err = [float(df_resullts['OR_MISMATCH_NO_CONGESTION'].value_counts()[1]*100/df_resullts.shape[0])]

print("OR PEAK MATCH", congested_nw)
print("OR VALLEY MATCH", uncongested_nw)

print("OR PEAK VALLEY MISMATCH", congestion_err)
print("OR VALLEY PEAK MATCH", uncongestion_err)

df2 = pd.DataFrame({'NETWORK_AT_PEAK': congested_nw, 'NETWORK_AT_VALLEY': uncongested_nw, 'PEAK_VALLEY_ERROR': congestion_err, 'VALLEY_PEAK_ERROR': uncongestion_err})
ax = df2.plot(kind='bar', stacked=True, figsize=(8, 6))


annonation_labels = [congested_nw, uncongested_nw, congestion_err, uncongestion_err]

# Add this loop to add the annotations
count = 0
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(str(round(p.get_y() + height + 0.01, 2)) + "%-(" + str(round(annonation_labels[count][0], 2)) + "%)", (p.get_x() + .5 * width, p.get_y() + height + 0.01), ha='center', fontsize=6)
    count = count + 1


plt.xticks(rotation=30)
plt.ylabel("Number Records - Actual vs Predicted Match/Mismatch", fontsize=10, labelpad=12)
plt.xlabel("Congestion Summary", fontsize=10, labelpad=12)
plt.title("Comparing OR Predictions - Actualvs Predicted(%)", fontsize=10, y=1.02)
plt.legend(loc='upper left', frameon=False, prop={'size': 6})
plt.savefig('figs/or.png')