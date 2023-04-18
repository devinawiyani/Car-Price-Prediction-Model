
#import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


# import dataset
df=pd.read_csv('Copy of Microsoft x Carsome Hackathon Auction Data Set - fancy-adventurous-peacock-62328.csv')


# replace for the missing data
df['car_variant']=df['car_variant'].fillna('no variant')
df['used_dealer_company_id']=df['used_dealer_company_id'].fillna(999999)

# remove null
df=df.dropna()


### Encode Categorical Data
#create a dictionary to determine ranking of brand, model, and variant
carbrand_transform= (df.groupby('car_brand')['reserveprice']
              .mean()
              .sort_values()
              .rank(method='dense', ascending=True)
              .astype(int)
              .to_dict())

carmodel_transform=(df.groupby('car_model')['reserveprice']
              .mean()
              .sort_values()
              .rank(method='dense', ascending=True)
              .astype(int)
              .to_dict())
carvariant_transform = (df.groupby('car_variant')['reserveprice']
              .mean()
              .sort_values()
              .rank(method='dense', ascending=True)
              .astype(int)
              .to_dict())

# Use frequency to encode the ids
lead_frequency_map=df['lead_id'].value_counts().to_dict()
mp_frequency_map=df['marketplace_id'].value_counts().to_dict()
mpc_frequency_map=df['marketplace_car_id'].value_counts().to_dict()
udc_frequency_map=df['used_dealer_company_id'].value_counts().to_dict()
dealer_frequency_map=df['dealer_id'].value_counts().to_dict()

# assign new  car brand, model, and variant
new_df=df.assign(car_brand=df['car_brand'].map(carbrand_transform), car_model=df['car_model'].map(carmodel_transform), car_variant=df['car_variant'].map(carvariant_transform))

# change auto to 0 and manual to 1
new_df['car_transmission'] = df['car_transmission'].replace({'Auto': 0, 'Manual': 1})
new_df.head()

# assign the ids from the encoding results
new_df['lead_id'], new_df['marketplace_id'], new_df['marketplace_car_id'], new_df['used_dealer_company_id'], new_df['dealer_id']=new_df['lead_id'].map(lead_frequency_map), new_df['marketplace_id'].map(mp_frequency_map), new_df['marketplace_car_id'].map(mpc_frequency_map), new_df['used_dealer_company_id'].map(udc_frequency_map), new_df['dealer_id'].map(dealer_frequency_map)

### Data Transformation
# check for skewness in data
new_df.hist(figsize= (20,15))

# Log Transform
new_df['lead_id']=np.log(new_df['lead_id'])
new_df['marketplace_id']=np.log(new_df['marketplace_id'])
new_df['marketplace_car_id']=np.log(new_df['marketplace_car_id'])
new_df['used_dealer_company_id']=np.log(new_df['used_dealer_company_id'])
new_df['dealer_id']=np.log(new_df['dealer_id'])
new_df['car_year']=np.log(new_df['car_year'])


## Data Split
# input variables
X= new_df.drop(['reserveprice'],axis = 1)

# target variable
y= new_df['reserveprice']

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


## Model Training
# train with random forest regressor

rf= RandomForestRegressor()
rf.fit(X_train, y_train)

# optimize model
from sklearn.model_selection import GridSearchCV

param_grid={
    "n_estimators": [30, 50,100],
    "max_features": [4],
    "min_samples_split": [2,4,8],
    "max_depth": [None, 4, 8]
    
}

grid_search = GridSearchCV(rf, param_grid, cv=5,
                          scoring ="neg_mean_squared_error",
                          return_train_score=True)

grid_search.fit(X_train, y_train)

best_rf=grid_search.best_estimator_


## Make pickle file
pickle.dump(best_rf, open("final-model.pkl", "wb"))

