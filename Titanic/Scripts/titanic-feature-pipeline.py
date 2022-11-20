import os
import modal

#Create titanic feature and upload to HopsWorks
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("lab1"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="test42")
    fs = project.get_feature_store()
    #read in data
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    #apply feature engineering to titanic datset
    drop=["PassengerId","Name","Ticket"]
    titanic_df.drop(drop,axis=1,inplace=True)
    #N is to indicate unknown
    titanic_df.Cabin.fillna("N",inplace=True)
    #get the first alphabet 
    titanic_df.Cabin=titanic_df.Cabin.apply(lambda x:x[0])
    titanic_df.rename(columns={"Cabin":"Deck"},inplace=True)
    titanic_df.Age.fillna(titanic_df.Age.median(),inplace=True)
    titanic_df.Embarked.fillna("N",inplace=True)
    titanic_df=pd.get_dummies(titanic_df)
    #rename columns to lower case for hopsworks feature store
    titanic_df.rename(columns={col:col.lower() for col in titanic_df.columns},inplace=True)

    #get or create feature group, need to modify name,primary key and description
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_feature_modal",
        version=1,
        #all columns except Survived, it is label/class
        primary_key=['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_female', 'sex_male',
       'deck_a', 'deck_b', 'deck_c', 'deck_d', 'deck_e', 'deck_f', 'deck_g',
       'deck_n', 'deck_t', 'embarked_c', 'embarked_n', 'embarked_q',
       'embarked_s'], 
        description="Titanic survivor dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
