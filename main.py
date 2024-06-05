import streamlit as st
import pandas as pd
from  sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pickle
import joblib
import math
df=pd.read_csv("df1.csv")
x=df.drop("runs_x",axis=1)
y=df["runs_x"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
trf=ColumnTransformer([("trf",OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore"),['batting_team','bowling_team','city'])],remainder="passthrough")
x_train.drop(columns='Unnamed: 0',axis=1,inplace=True)
x_train=trf.fit_transform(x_train)
pipe=Pipeline(steps=[("step1",StandardScaler()),("step2",RandomForestRegressor(n_estimators=10,max_depth=22
                                                                               ,random_state=1))])
pipe.fit(x_train,y_train)
venue=df["city"].unique().tolist()
teams=df["batting_team"].unique().tolist()
venue.sort()
teams.sort()
st.title("T20 Score PREDICTOR")
col1,col2=st.columns(2)
with col1:batting_team=st.selectbox("Select the Batting team",teams)
with col2:bowling_team=st.selectbox("Select the Bowling team",teams)
selected_city=st.selectbox("Select Venue",venue)
col3,col4,col5,col6=st.columns(4)
with col3:score=st.number_input('Score',step=1)
with col4:balls=st.number_input('Balls completed(Valid Balls)',step=1,min_value=30,max_value=120)
with col5:last_five=st.number_input("Last 5 over runs",step=1)
with col6:wickets=st.number_input('Wickets Out',step=1)
if st.button("Predict Score"):
    balls_left=120-balls
    wickets=10-wickets
    crr=(score/balls)*6
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team], 'city':[selected_city],'current_score':[score],
    'balls_left':[balls_left], 'wickets_left':[wickets],'crr':[crr], 'last_five':[last_five]})
    st.table(input_df)
    x=trf.transform(input_df)
    result=pipe.predict(x)[0]
    st.subheader("Predicted Score of " + batting_team + "- " + str(math.ceil(result)))


