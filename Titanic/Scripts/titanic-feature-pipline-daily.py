import os
import modal
    
BACKFILL=False
LOCAL=False

if LOCAL == False:
   stub = modal.Stub("titanic_feature")
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("lab1"))
   def f():
       g()


def generate_passenger(survived,
                    pclass_min,pclass_max,
                    age_min,age_max,
                    sibsp_min,sibsp_max,
                    parch_min,parch_max,
                    fare_min,fare_max,
                    deck_weights,
                    embarked_weights):
    import pandas as pd
    import random
    sex_female=0
    deck=[i for i in "abcdefgnt"]
    deck_dict={c:0 for c in "abcdefgnt"}
    embarked=[i for i in "cnqs"]
    embarked_dict={c:0 for c in "cnqs"}
    
    if survived==1:
       if random.random()<0.7:
          sex_female=1
       

    else:
       if random.random()<0.15:
          sex_female=1
    sex_male=1-sex_female
    deck_dict[random.choices(deck,deck_weights,k=1)[0]]=1
    embarked_dict[random.choices(embarked,embarked_weights,k=1)[0]]=1
    #generate random attributes
    df = pd.DataFrame({ "pclass": [random.randint(pclass_min, pclass_max)],
                        "age": [random.uniform(age_min, age_max)],
                       "sibsp": [random.randint(sibsp_min, sibsp_max)],
                       "parch": [random.randint(parch_min, parch_max)],
                       "fare": [random.uniform(fare_min, fare_max)],
                       'sex_female':[sex_female],
                       'sex_male':[sex_male],
                       'deck_a':deck_dict["a"],
                       'deck_b':deck_dict["b"],
                       'deck_c':deck_dict["c"],
                       'deck_d':deck_dict["d"],
                       'deck_e':deck_dict["e"],
                       'deck_f':deck_dict["f"],
                       'deck_g':deck_dict["g"],
                       'deck_n':deck_dict["n"],
                       'deck_t':deck_dict["t"],
                       'embarked_c':embarked_dict["c"],
                       'embarked_n':embarked_dict["n"],
                       'embarked_q':embarked_dict["q"],
                       'embarked_s':embarked_dict["s"]
                      })
    df['survived'] = survived
    convert={
                       'sex_female':"uint8",
                       'sex_male':"uint8",
                       'deck_a':"uint8",
                       'deck_b':"uint8",
                       'deck_c':"uint8",
                       'deck_d':"uint8",
                       'deck_e':"uint8",
                       'deck_f':"uint8",
                       'deck_g':"uint8",
                       'deck_n':"uint8",
                       'deck_t':"uint8",
                       'embarked_c':"uint8",
                       'embarked_n':"uint8",
                       'embarked_q':"uint8",
                       'embarked_s':"uint8"
       }
    df=df.astype(convert)
    return df


def get_random_passenger():
   import pandas as pd
   import random
   survived = generate_passenger(1,
                                  1,2,
                                  0,80,
                                  0,8,
                                  0,6,
                                  50,550,
                                  [2,10,10,7,7,2,0,60,0],
                                  [27,0,8,63])
   dead = generate_passenger(0,
                              2,3,
                              0,80,
                              0,8,
                              0,6,
                              0,100,
                              [1,2,4,1,1,0,0,87,0],
                              [13,0,8,77])
   if random.random()<0.4:
      #survive
      res=survived
      print("Added survived passenger")
   else:
      res=dead
      print("Added dead passenger")
   return res




def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="test42")
    fs = project.get_feature_store()

    if BACKFILL == True:
        df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        df = get_random_passenger()

    fg = fs.get_or_create_feature_group(
        name="titanic_feature_modal",
        version=1,
        primary_key=['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_female', 'sex_male',
       'deck_a', 'deck_b', 'deck_c', 'deck_d', 'deck_e', 'deck_f', 'deck_g',
       'deck_n', 'deck_t', 'embarked_c', 'embarked_n', 'embarked_q',
       'embarked_s'], 
        description="Titanic survivor dataset")
    fg.insert(df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
       stub.deploy("titanic_feature")
       with stub.run():
            f()
