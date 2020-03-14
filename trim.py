import pandas as pd
df = pd.read_csv('/Users/dyuwan/Desktop/ARDOS Datasets/songdata.csv')
check= df['mood']==0
df_0=df[check]
df_0.to_csv('/Users/dyuwan/Downloads/happysongs.csv')
def song():
    k=df.sample(1)[['artist','song', 'link']]
    s=k.to_dict('list')
    return s
