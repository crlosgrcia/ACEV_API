# import pandas as pd
# import numpy as np
# import datetime
# import pickle
# from create_inperf import api
# import os
# from scipy.special import expit

# Method Loader for the API

#########################################################################################
# Creates a global variable to fix vehicles in payload
# payload goes into fixvehicles first to add default values to the vehicles

global fixvehicles
        
# Define a fixvehicles method that takes in a payload and adds default values to the vehicles
# fixvehicles() further processes the payload from pydantic validation by assigning more default values to potentially missing keys and transforming key cases.

def fixvehicles(payload):
    # create default start and end month as current month + 5 years
    curmonth = datetime.date.today()
    start_month = curmonth - datetime.timedelta(days=curmonth.day-1)
    if start_month.month == 1:
        end_month = datetime.date(start_month.year + 30,12,1)
    else:
        end_month = datetime.date(start_month.year+ 30, start_month.month, 1)
    print(start_month,end_month)
    
    # Defines default values for vehicles in payload
    defaultvehicle = {
        "sale_type":"Unknown",
        "moodys_region":"South",
        "interior_color":".",
        "exterior_color":".",
        "start_month":str(start_month),
        "end_month":str(end_month)
    }
    # Defines missing default columns
    missing_default_columns = ['liters', 'cylinders', 'drive_type', 
                            'body_type', 'segment', 'fuel_type',
                'induction_type','make', 'trim_level', 'msrp', 'model_year']
    
    # Adds default values to vehicles in payload, will be supplemented by lookup data in def forecast
    for c in missing_default_columns:
        defaultvehicle[c] = ''
    
    # Creates a unique VID for each vehicle in payload based on lower case keys
    # and removes model_year to avoid potential conflicts
    # and adds default values to vehicles in payload
    for i in range(0,len(payload['vehicles'])):
        c = dict((k.lower(),v) for k,v in payload['vehicles'][i].items())
        c['VID'] = i
        for k,v in defaultvehicle.items():
            c.setdefault(k,v)
        # Updates vehicles in payload with default values
        payload['vehicles'][i] = c
    
# Define a forecast method that takes in data, query, and payload and returns a dataframe
def forecast(api:dict,query:dict,payload:dict): # data is the api dictionary, query is the query string parameters, and payload is the request body
    # for example, 
    # Call global function to fix vehicles in payload
    fixvehicles(payload)
    
    # Get vehicle ids and create a dataframe
    vids = {x['VID'] for x in payload['vehicles']}
    vins = pd.DataFrame.from_dict(payload['vehicles'])
    scen = payload['scenario']
    
    # Create 10 digit VIN from payload
    vins['short_vin'] = [f'{v[0:8]}_{v[9]}'.upper() for v in vins['vin'].values]
    
    # # Open the API pickle with the fitted model, rhs, and lookup table, and Fecon data
    # with open(api, "rb") as fp:
        
    #     api = pickle.load(fp)
        
    # Merge Payload Vins and lookup dataframe # This is inner so that only the vins that are in both are kept
    
    vins = vins.merge(api['lookup_no_trim'], on='short_vin', how='inner')
    
    # Get the VID's that were not found in the lookup table
    notfound = vids - set(vins['VID'])
    for vid in notfound:
        payload['vehicles'][vid]['error'] = True
        payload['vehicles'][vid]['message'] = 'VIN was not found'
        
    # Fill in missing values from payload with lookup values
    lookup_columns = ['liters', 'cylinders', 'drive_type', 'body_type', 'segment', 'fuel_type',
                    'induction_type', 'make', 'trim_level', 'msrp', 'model_year']
    for x in lookup_columns:
        vins[x] = np.where(vins[x]=='', vins['mod_'+x+''], vins[x])
        vins[x] = np.where(vins[x]=='.', vins['mod_'+x+''], vins[x])
        vins[x] = vins[x].astype(vins['mod_'+x].dtype)
    
    # work out the valid date range, if any, based on model_year
    vins['start_month'] = pd.to_datetime(vins['start_month'])
    vins['end_month'] = pd.to_datetime(vins['end_month'])
    
    # print(vins['start_month']) # 2024-07-01
    # print(vins['end_month']) # 2029-06-01
    
    vins['vintage_month'] = (vins['model_year']-1).astype(str)+'-'+'01'+'-'+'01'
    vins['vintage_month'] = pd.to_datetime(vins['vintage_month'])
    
    # Choose the latest of the vintage month and the start month
    vins['start_month'] = vins[['start_month','vintage_month']].max(axis=1)
    
    
    # Not sure what the purpose of this was supposed to be as then any vehicle with a model year before 2017 would be excluded in next few steps down
    # # Create a max end date based on model year + 7 years
    # vins['max_end'] = (vins['model_year']+7).astype(str)+'-'+'01'+'-'+'01'
    # # Convert max end date to datetime
    # vins['max_end'] = pd.to_datetime(vins['max_end'])
    # # Choose the earliest of the max end date and the end month
    # vins['end_month'] = vins[['end_month','max_end']].min(axis=1)
    # Drop max end date
    # vins.drop('max_end',axis=1,inplace=True)
    
    # print(vins['start_month'])
    # print(vins['end_month'])

    # Dummy for model 
    vins['vehicle_type'] = "Light Truck"
    vins['car'] = vins['segment'].str.contains("Car").astype(int)
    vins.loc[vins['car'] == 1, 'vehicle_type'] = "Car"
    vins = vins.drop(columns=['car'])
    
    # generating truck and car dummies for interaction terms
    vins['segment'] = vins['segment'].astype(str) 
    vins['luxury'] = vins['segment'].apply(lambda x: 1 if 'luxury' in x.lower() or 'premium' in x.lower() else 0)
    vins['truck'] = vins['segment'].apply(lambda x: 1 if 'pickup' in x.lower()
                                    or 'truck' in x.lower()
                                    or 'conventional' in x.lower()
                                    or 'utility' in x.lower()
                                    or 'suv' in x.lower()
                                    or 'van' in x.lower()
                                    else 0)

    # Create a car variable
    vins['car'] = 1 - vins['truck']
    
    # Create a log trim variable
    
    vins['log_trim_level'] = np.where((vins['trim_level'].notna()) & (vins['trim_level'] > 0), np.log(vins['trim_level']), vins['trim_level']) 
    
    ########################### These categories can be dummified because they don't change often ############################
    
    # Incoming vin example: {'VIN':'3GB5YTE7_M','initial_mileage':34601,'annual_mileage_assumption':12000,
    # 'msrp':140009,'moodys_region':'South'
    # 'interior_color':'.','sale_type':'As Is','model_year':2019},
    
    # Only have to do this here for the moodys_region because it is a categorical variable that gets used as an interaction
    
    moodys_regions = ['Northeast','South','Midwest','West']
    
    categorical_vars = ['moodys_region']

    for cat_var in categorical_vars:
        # Generate dummies from existing categories
        dummy_df = pd.get_dummies(vins[cat_var], prefix=cat_var)
        
        # Add other missing dummy columns if any with 0s 
        expected_cols = [f"{cat_var}_{c}" for c in moodys_regions]
        
        missing_cols = set(expected_cols) - set(dummy_df.columns)
        
        for col in missing_cols:
            dummy_df[col] = 0    

        vins.drop(cat_var, axis=1, inplace=True)
        
        # Append generated/complete dummy dataframe with main dataframe  
        vins = pd.concat([vins, dummy_df], axis=1)
    
    # baseline_level_vars = ["make_ALFA ROMEO", "moodys_region_Midwest", "segment_Compact Car",
    #             "liters_0.0", "cylinders_0.0","drive_type_4WD","body_type_Cargo Van",
    #             "sale_type_As Is","fuel_type_CNG","induction_type_Standard",
    #             "exterior_color_.",  "interior_color_.",  "transmission_A",
    #             "vehicle_type_Car", "int_age_year_0"]
    
    ##############################################################################################################
    
    # make all incoming makes uppercase to match model input data
    
    vins['make'] = vins['make'].str.upper()
    
    #exclude vehicles that are too old from payload, this drops any vehicle with a model year before 2017, because start month is forecasted to be 2024-07-01
    # vids = set(vins['VID'])
    # vins = vins[vins.end_month >= vins.start_month]
    # excluded = vids - set(vins['VID'])
    
    # for vid in excluded:
    #     payload['vehicles'][vid]['error'] = True
    #     payload['vehicles'][vid]['message'] = 'Vehicle was too old'
        
    # # exclude vehicles with bad input parameters, 
    # if any in test_vins.csv is empty they will get dropped, for example, transmission is causeing the current vin to drop
    # vins['badparams'] = vins.isna().sum(axis=1) > 0
    # badparams = set(vins[vins['badparams']].VID)
    
    # for vid in badparams:
    #     payload['vehicles'][vid]['error'] = True
    #     badfields = str.join(', ', vins.columns[vins[vins.VID == vid].isna().sum(axis=0).astype(bool)].to_list())
    #     payload['vehicles'][vid]['message'] = f'Invalid value in fields : ({badfields})'

    # # remove vehicles with bad parameters
    # vins = vins[vins['badparams'] == False]
    # vins.drop('badparams', axis=1,inplace=True)
    
    # create a table with range per VID
    range_table = vins[['VID','start_month','end_month']]
    
    # get beginning of date range
    begin_fcast = range_table['start_month'].min()

    # Print
    
    # print(begin_fcast)
    
    # convert to datetime
    
    begin_fcast = pd.to_datetime(begin_fcast, format='%Y-%m')
    
    # get end of date range
    
    end_fcast = range_table['end_month'].max()
    
    # # Print
    # print(end_fcast)
    
    # convert to datetime format month year
    
    end_fcast = pd.to_datetime(end_fcast, format='%Y-%m')
    
    ############################## create date table
    date_table = pd.DataFrame()
    ############################## create mtime column
    date_table['mtime'] = pd.date_range(begin_fcast,end_fcast, freq='MS') 
    #  'mtime' is added to the date_table, 
    # which represents every month starting from begin_fcast (the earliest start month) up to end_fcast (the latest end month), 
    # as specified by pandas date_range function with frequency parameter set as 'MS', standing for Month Start frequency. 
    
    # merge date table with range table
    df = range_table.assign(key=0).merge(date_table.assign(key=0),how='left', on='key')
    
    df = df[(df['mtime'] >= df['start_month']) & (df['mtime'] <= df['end_month'])]
    
    df = df[['VID','mtime']]
    
    df = vins.merge(df,how='left',on='VID')
    
    # update remaining payload records
    vids = set(vins['VID'])

    # set dates back to strings (in case they were changed by mage limits)
    vins['start_month'] = [str(x.date()) for x in vins['start_month']]
    vins['end_month'] = [str(x.date()) for x in vins['end_month']]


    # Calculate mage - propogates changes forward for forecasting
    #df['mage'] = round((df['mtime'] - df['vintage_month'])/np.timedelta64(1, 'M'))

    # print(df["mtime"].dtype)

    # print(df["vintage_month"].dtype)

    from pandas.tseries.offsets import DateOffset
    df['mage'] = df.apply(lambda row: len(pd.date_range(start=row.vintage_month, end=row.mtime, freq='ME')), axis=1)
    
    # convert to int
    # df['mage'] = df['mage'].astype(int)
    
    # Accumulate miles - this propogates mileage forward
    # df['mileage'] = df['initial_mileage'] + (df['annual_mileage_assumption']/12)*round((df['mtime'] - df['start_month'])/np.timedelta64(1, 'M'))

    # Calculate the number of months between 'start_month' and 'mtime'
    df['months_diff'] = df.apply(lambda row: len(pd.date_range(start=row['start_month'], end=row['mtime'], freq='ME')), axis=1)

    # Calculate mileage using the months difference
    df['mileage'] = df['initial_mileage'] + (df['annual_mileage_assumption'] / 12) * df['months_diff']

    
    # Create yearly age dummy for interaction with mileage spline
    
    df["int_age_year"] = pd.DatetimeIndex(df["mtime"]).year - df["model_year"] + 1
    
    int_age_dummy = ["int_age_year"]
    
    for dum in int_age_dummy:
        dummies = pd.get_dummies(df[dum], prefix=dum)
        df = pd.concat([df, dummies], axis=1)
        
    # Get the names of the newly created dummy columns
    age_dummies = [col for col in df.columns if 'int_age_year' in col]
    
    # reset df index
    
    df.reset_index(drop=True, inplace=True)
    
    ###################### Create mileage spline
    
    # Define mileage variable
    mileage = df["mileage"]

    # Define your knots here
    knots = [1000, 6000, 30000, 60000, 120000, 200000]

    def rcs(x, knots):
        df_splines = pd.DataFrame()
        
        df_splines[f'spline_1'] = x.copy()

        # Define first (k_1) - minimum and last (k_n) - maximum knots
        k_1 = knots[0]
        k_n = knots[-1]
        
        def d(x, k): return ((x-k)**3)*(x>k)

        # Generate splines 2 to n
        for i in range(1,len(knots)):
            
            knot_i = knots[i-1]
            
            temp_col=d(x, knot_i) 
            
            if i<len(knots): # Exclude top-most knot value
                next_k=knots[i]            
                temp_col -= ((d(x,k_n)-d(x,next_k))*(k_n-next_k))/((k_n-k_1)*(k_n-knot_i))
                
            df_splines[f'spline_{i+1}'] = temp_col / ((k_n - k_1)**2)

        return df_splines.values

    # Call the function on your specific data and save the result
    mileage_splines = rcs(mileage,knots)

    # Convert this array into pandas DataFrame with appropriate column names
    splines_df = pd.DataFrame(mileage_splines, columns=[f'spline_{i}' for i in range(1,len(knots)+1)])

    # Drop last spline by position in the dataframe

    splines_df = splines_df.drop(splines_df.columns[-1], axis=1)
    
    # reset spline_df index
    
    splines_df.reset_index(drop=True, inplace=True)
    
    # Concatenate splines_df with df
    
    df = pd.concat([df, splines_df], axis=1)
    
    # Create mileage and model year interaction
    
    # get the names of the spline variables

    spline_variable = df.filter(like='spline')

    spline_variable.reset_index(drop=True, inplace=True)

    # For each age dummy variable, interact it with the spline variables
    for age_col in age_dummies:
        for spline_col in spline_variable.columns:
            df[f'{age_col}_{spline_col}'] = df[age_col] * spline_variable[spline_col]
            if df[f'{age_col}_{spline_col}'].isna().any():
                print(f'NaNs in interaction term {age_col}_{spline_col}')
                print(df[f'{age_col}_{spline_col}'].isna().sum())
    
    # Define the categorical variables
    
    # categorical_vars = ['int_age_year']
    
    # baseline_level_vars = ["int_age_year_0"] # is done here after mileage is propogated forward because the rest of these stay the same throughout the forecast
    
    # # Convert incoming categorical variables to dummies
    # for var in categorical_vars:
        
    #     if var in vins.columns:
    #         dummies = pd.get_dummies(df[var], prefix=var).astype(float)
            
    #         # Drop column related to original variable 
    #         df.drop(var, axis=1, inplace=True)
            
    #         for col in dummies.columns:
    #             if col not in baseline_level_vars:  # Keep if not baseline level
    #                 df[col] = dummies[col]
    
    # Keep if vehicle sold on or after vintage month
    df = df[(df['mtime']>=df['vintage_month'])]
    
    ### Vehicle feature engineering ###

    # Create seasonality variable
    df['tseas'] = df['mtime'].dt.month
    
    df = pd.get_dummies(df, columns=['tseas'], drop_first=True)

    df.rename(columns={'tseas_2.0': 'tseas_2', 'tseas_3.0': 'tseas_3', 'tseas_4.0': 'tseas_4', 'tseas_5.0': 'tseas_5', 'tseas_6.0': 'tseas_6', 
            'tseas_7.0': 'tseas_7', 'tseas_8.0': 'tseas_8', 'tseas_9.0': 'tseas_9', 'tseas_10.0': 'tseas_10', 'tseas_11.0': 'tseas_11', 'tseas_12.0': 'tseas_12'}, inplace=True)

    tseas_dummies = [col for col in df.columns if col.startswith('tseas')]

    ################### Convert tseas dummies to uint8 #######################
    for col in df.columns:
        if col.startswith('tseas'):
            df[col] = df[col].astype(np.uint8)
        
    ################### Create interaction variables for 'tseas' and 'moodys_region' #######################

    # List of moodys_region columns, midwest is the baseline level
    moodys_region_columns = ['moodys_region_Northeast', 'moodys_region_South', 'moodys_region_West']

    # Initialize an empty list to hold the names of the newly created interaction columns
    interaction_columns = []

    # For each 'tseas' dummy variable, create an interaction term with each 'moodys_region' column
    for dummy in tseas_dummies:
        for region in moodys_region_columns:
            interaction_column_name = f'{dummy}_{region}'
            df[interaction_column_name] = df[dummy] * df[region]
            interaction_columns.append(interaction_column_name)
            
    # # Generate qvintage variable
    # df['qvintage'] = df['model_year'].astype(str)+'-'+'10'+'-'+'01'
    # # Convert df['qvintage'] from object to date format
    # df['qvintage'] = pd.to_datetime(df['qvintage'])
    
    ### Import tecon data and perform feature engineering ###
    
    # Merge df and tecon dataframes 
    
    df = pd.merge(df, api[f'tecon_{scen}'], on='mtime')
    
    # # Send to csv
    
    # df.to_csv(fr'O:\autocycle\AC_EV\API\df_with_tecon_data.csv', index=False)
    
    # Merge df and vecon dataframes 
    
    df = pd.merge(left = df, right = api[f'vecon_{scen}'], how = 'left' , left_on = 'vintage_month', right_on = 'mtime') 
    
    # drop mtime_y
    
    df = df.drop('mtime_y', axis=1)
    
    # rename mtime_x to mtime
    
    df.rename(columns={'mtime_x': 'mtime'}, inplace=True)
    
    # # Send to csv
    
    # df.to_csv(fr'O:\autocycle\AC_EV\API\df_with_vecon_data.csv', index=False)
    
    # Vecon feature engineering
    
    df['vtxxyp_gasdeflate'] = ((df['txx_gasdeflate']/df['vxx_gasdeflate'])-1)*100
    
    # Car Index 
    
    df['txx_fvhirncaq'] = df['txx_fvhirncaq'].replace('ND', np.nan).astype(float).dropna()
    df['vxx_fvhirncaq'] = df['vxx_fvhirncaq'].replace('ND', np.nan).astype(float).dropna()
    df['vtxxyp_fvhirncaq'] = ((df['txx_fvhirncaq']/df['vxx_fvhirncaq']) - 1) * 100
    
    # Truck Index
    df['txx_fvhirntaq'] = df['txx_fvhirntaq'].replace('ND', np.nan).astype(float).dropna()
    df['vxx_fvhirntaq'] = df['vxx_fvhirntaq'].replace('ND', np.nan).astype(float).dropna()
    df['vtxxyp_fvhirntaq'] = ((df['txx_fvhirntaq']/df['vxx_fvhirntaq']) - 1) * 100
    
    # # EV Index
    
    # df['txx_fvhirneleaq'] = df['txx_fvhirneleaq'].replace('ND', np.nan).astype(float).dropna()
    # df['vxx_fvhirneleaq'] = df['vxx_fvhirneleaq'].replace('ND', np.nan).astype(float).dropna()
    # df['vtxxyp_fvhirneleaq'] = ((df['txx_fvhirneleaq']/df['vxx_fvhirneleaq']) - 1) * 100
    
    # # Create Other Ev index transformation variables

    # df['vtxxtdlog_fvhirneleaq'] = ((df['txxtdlog_fvhirneleaq']/df['vxxtdlog_fvhirneleaq']) - 1) * 100
    
    # # Create lagged variables
    # df['txxyp_frprime_3lag'] = df['txxyp_frprime'].shift(3)
    
    #segment_mage_squared
    df['segment_mage_squared'] = df['segment_main'] * df['mage']**2	
    df['mage_squared'] = df['mage']**2
    # Interact inventories with industrial production
    df['fip335_txx_fscard'] = df['txx_fip335'] * df['txx_fscard']
    # vxxtdlog_fscard and txxtdlog_fscard
    df['vtxxtdlog_fscard'] = ((df['txxtdlog_fscard']/df['vxxtdlog_fscard']) - 1) * 100
    # Segment Gasoline Deflator Mix
    df['segment_txx_gasdeflate'] = df['segment_main'] * df['txx_gasdeflate']
    df['txx_elecdeflate'] = df['txx_fcpiuehf']/df['txx_fcpiu']
    df['segment_txx_elecdeflate'] = df['segment_main'] * df['txx_elecdeflate']
    # # Car Electricity Deflator Mix
    # df['car_txx_electdeflate'] = df['car'] * df['txx_elecdeflate']
    # df['truck_txx_electdeflate'] = df['truck'] * df['txx_elecdeflate']
    # # EV Transformed Index Mix
    # df['car_vtxxyp_fvhirneleaq'] = df['car'] * df['vtxxyp_fvhirneleaq']
    # df['truck_vtxxyp_fvhirneleaq'] = df['truck'] * df['vtxxyp_fvhirneleaq']
    # # EV Log Transformed Index Mix
    # df['car_vtxxtdlog_fvhirneleaq'] = df['car'] * df['vtxxtdlog_fvhirneleaq']
    # df['truck_vtxxtdlog_fvhirneleaq'] = df['truck'] * df['vtxxtdlog_fvhirneleaq']
    # # EV Index Mix
    # df['car_txx_fvhirneleaq'] = df['car'] * df['txx_fvhirneleaq']
    # df['truck_txx_fvhirneleaq'] = df['truck'] * df['txx_fvhirneleaq']
    # # Disposable Income Mix
    # df['car_txx_fypdpiq'] = df['car'] * df['txx_fypdpiq']
    # df['truck_txx_fypdpiq'] = df['truck'] * df['txx_fypdpiq']
    # Segment unemployment mix
    df['segment_txx_flbr'] = df['segment_main'] * df['txx_flbr']
    
    # drop vintage month
    df = df.drop('vintage_month', axis=1)
    
    # Keep columns needed for analysis
    vin = df['vin']
    VID = df['VID']
    msrp = df['msrp']
    mod_msrp = df['mod_msrp']
    
    # Initialize a new column 'old_ev_ind' with 0       
    df['old_ev_ind'] = 0

    # Create 'old_ev_ind' to 1 if 'model_year' is less than or equal to 2017 and 'fuel_type' is 3 or 10

    df.loc[(df['model_year'] <= 2017) & ((df['fuel_type']== 'Electric') | (df['fuel_type']== 'Plug-in Hybrid')), 'old_ev_ind'] = 1
    
    # put mtime on the far left hand side
    
    df = df[['mtime'] + [col for col in df.columns if col != 'mtime']]
    
    ################ Used fitted model to predict new values based on input dataframe, as long as it contains the same columns as the original dataframe used to fit the model ################
    # Load the saved model/features-schema
    # with open(api,"rb") as fp: 

    #     api=pickle.load(fp)

    # Extract existing feature names from loaded DF/model
    existing_features = list(api['rhs'].columns)

    # List of columns to be dummified
    columns_to_dummify = ['make', 'segment', 'liters', 'cylinders',
                        'drive_type', 'body_type', 'sale_type','fuel_type',
                        'induction_type','exterior_color',
                        'interior_color','transmission', 'vehicle_type']

    # Apply pd.get_dummies only on selected columns
    df_dummified = pd.get_dummies(df, columns=columns_to_dummify)

    # Identify missing columns in df (after dummyfying) that exist in original trained features-schema 
    missing_cols = set(existing_features) - set(df_dummified.columns)
    for col in missing_cols:
        df_dummified[col] = 0

    # Now vins_dummified has same columns/features as your original df upon which model was trained.
    
    # Access the fitted model
    fitted_model = api['model']
    
    # print(fitted_model.summary())   
    
    # Create the RHS variables
    rhs = api['rhs']
    
    # # # send rhs columns to csv
    
    # pd.DataFrame(rhs).to_csv(fr'O:\autocycle\AC_EV\df_rhs.csv', index=False)
    
    # send df_dummified columns to csv
    
    # pd.DataFrame(df_dummified.columns).to_csv(fr'O:\autocycle\AC_EV\df_dummified_columns.csv', index=False)
    
    # print(df_dummified.head())
    # print(df_dummified.info())
    
    # Create the X variables
    X = df_dummified[rhs.columns]
    
    # send X columns to csv
    
    # pd.DataFrame(X).to_csv(fr'O:\autocycle\AC_EV\X.csv', index=False)
    
    # Check for NaN values
    
    # NaN_df = pd.concat({i: X[i][X[i].isnull()] for i in X.columns if X[i].isnull().any()})
    # NaN_df.to_csv(fr'O:\autocycle\AC_EV\nan_values.csv', index=False)
    
    # print(X.isnull().sum())  # This should return 0 for all columns.
    # print(np.isinf(X).sum()) # This should return 0 for all columns.
    
    # Make predictions
    df_dummified['flogit_xprice'] = fitted_model.predict(X)
    
    ##############################################################################################################

    df_dummified['flogit_xprice'] = pd.to_numeric(df_dummified['flogit_xprice'], errors='coerce')
    
    # add model_calib to flogit_xprice
    df_dummified['flogit_xprice'] = df_dummified['flogit_xprice'] + df_dummified['model_calib']
    
    # Forecast Price
    df_dummified['fxprice'] = expit(df_dummified['flogit_xprice']) #Residual Value (Fitted)
    
    # add vqi_adj
    df_dummified['fxprice'] = df_dummified['fxprice'] + df_dummified['vqi_adj']
    
    # Merge dropped data back in
    df_dummified['VID'] = VID
    df_dummified['vin'] = vin
    df_dummified['msrp'] = msrp
    df_dummified['mod_msrp'] = mod_msrp
    # Use mod_msrp if msrp is missing
    df_dummified['msrp'] = df_dummified['msrp'].astype('category')
    df_dummified['msrp'] = np.where(df_dummified['msrp']=='', df_dummified['mod_msrp'], df_dummified['msrp'])
    # Convert msrp to float
    df_dummified['msrp'] = pd.to_numeric(df_dummified['msrp'], errors='coerce')
    # Calculate fprice
    df_dummified['fprice'] = df_dummified['fxprice']*df_dummified['msrp']
    
    # Add Calibration and VQI values from no trim look up table
    
    # keep only the VID, vin, and forecasted price columns
    df_dummified = df_dummified[['mtime', 'VID','vin','fprice','fxprice', 'mileage']]
    
    # Eliminates duplicates of VID
    succvids = set(df_dummified['VID'])
    for vid in succvids:
        payload['vehicles'][vid]['forecastPrice'] = [round(x,2) for x in df_dummified['fprice'][df_dummified['VID'] == vid].to_list()]
        payload['vehicles'][vid]['forecastMileage'] = [round(x,0) for x in df_dummified['mileage'][df_dummified['VID'] == vid].to_list()]
    return df_dummified

methods['post']['forecast'] = forecast

# print(methods) # {'get': {}, 'post': {'forecast': <function forecast at 0x000001B7C324A020>}, 'loader': <code object <module> at 0x000001B7C1976E40, file "method_loader", line 1>}

#########################################################################################

# print(methods) # success, forecast method added to methods dictionary
# {'get': {}, 'post': {'forecast': <function forecast at 0x000002277AAFC670>}, 'loader': <code object <module> at 0x00000227AB773870, file "method_loader", line 1>}
# methods is a nested dictionary where the outer dictionary has HTTP methods (e.g., 'post') as keys and each value is another dictionary like forecast and loader.







