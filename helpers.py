from datetime import timedelta

#convert strings representing timedeltas to seconds
def seconder(x):
    mins, secs = map(float, str(x).split(':'))
    td = timedelta(minutes=mins, seconds=secs)
    return td.total_seconds()

def format_pts_mins(df):
    df['MIN'] = df['MIN'].astype('string')
    df['MIN'] = (df['MIN'].apply(seconder) / 60).astype(int)
    df['PTS'] = df['PTS'].astype(int)