import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle
from werkzeug.utils import secure_filename

# Create flask app
app = Flask (__name__)

# Pickle model
model=pickle.load(open("final-model.pkl","rb"))

# specify allowed format
ALLOWED_EXTENSIONS=set({'csv'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# preprocess retrieved file
def preprocess(df):
    dfs=pd.read_excel("output.xlsx", sheet_name=None)

    #dfs = {sheet_name: data_detail.parse(sheet_name) for sheet_name in data_detail.sheet_name}

    # read dataframes to dictionary
    carbrand_transform=dfs['car_brand'].set_index('car_brand')['car_brand_rank'].to_dict()
    carmodel_transform=dfs['car_model'].set_index('car_model')['car_model_rank'].to_dict()
    carvariant_transform=dfs['car_variant'].set_index('car_variant')['car_variant_rank'].to_dict()
    lead_frequency_map=dfs['lead'].set_index('lead')['lead_frequency'].to_dict()
    mp_frequency_map=dfs['mp'].set_index('mp')['mp_frequency'].to_dict()
    mpc_frequency_map=dfs['mpc'].set_index('mpc')['mpc_frequency'].to_dict()
    udc_frequency_map=dfs['udc'].set_index('udc')['udc_frequency'].to_dict()
    dealer_frequency_map=dfs['dealer'].set_index('dealer')['dealer_frequency'].to_dict()
    # assign new car brand, model, and variant

    new_df=df.assign(car_brand=df['car_brand'].map(carbrand_transform), car_model=df['car_model'].map(carmodel_transform), car_variant=df['car_variant'].map(carvariant_transform))

    # change auto to 0 and manual to 1
    new_df['car_transmission'] = df['car_transmission'].replace({'Auto': 0, 'Manual': 1})

    # assign the ids from the encoding results
    new_df['lead_id'], new_df['marketplace_id'], new_df['marketplace_car_id'], new_df['used_dealer_company_id'], new_df['dealer_id']= new_df['lead_id'].map(lead_frequency_map), new_df['marketplace_id'].map(mp_frequency_map), new_df['marketplace_car_id'].map(mpc_frequency_map), new_df['used_dealer_company_id'].map(udc_frequency_map), new_df['dealer_id'].map(dealer_frequency_map)

    # log transform
    new_df['lead_id']=np.log(new_df['lead_id'])
    new_df['marketplace_id']=np.log(new_df['marketplace_id'])
    new_df['marketplace_car_id']=np.log(new_df['marketplace_car_id'])
    new_df['used_dealer_company_id']=np.log(new_df['used_dealer_company_id'])
    new_df['dealer_id']=np.log(new_df['dealer_id'])
    new_df['car_year']=np.log(new_df['car_year'])

    return new_df

# prediction function
def prediction(row):
    return model.predict([row])[0]

# default page
@app.route("/")
def Page():
    return render_template("index.html")

# predict page
@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method =='POST':
        file=request.files['csvfile']
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)

            df=pd.read_csv(file)
            new_df=preprocess(df)
            new_df['reserveprice']=new_df.apply(prediction, axis=1)
            new_df['reserveprice']=new_df['reserveprice'].astype(int)
            new_filename=f'{filename.split(".")[0]}_{"output"}.csv'
            new_df.to_csv(os.path.join('Output', new_filename), index=False)

        return 'Prediction Successful!'
    return render_template('index.html')

# run app
if __name__ == "__main__":
    app.run(debug=True)

        





