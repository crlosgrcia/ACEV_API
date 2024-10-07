import pickle as pickle
import json
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import shutil


# The create_inperf function creates the api and methods objects for the inperf API Lambda based on ver and scenlist. 
# The api object contains the model and lookup table data, while the methods object contains the loader method. 
# The function loads the model results for the specified version and scenario list, and creates the inperf API object

def create_inperf(ver,scenlist):
    # create api object with all necessary data
    api = {}
    
    # create methods object for the api
    methods = {"get":{},"post":{}}

    # load model version - Model results for 072024 saved to O:/autocycle/AC_EV/model_dev/2023/dev_output/reestimate_forecast/model_55/072024model_55.pkl
    # 'v202407' = ver
    with open(fr"O:\autocycle\AC_EV\model_dev\2023\dev_output\reestimate_forecast\model_55\{ver}model_55.pkl","rb") as fp:
        
        loaded_data= pd.read_pickle(fp)
        
        # The error message suggests the pickle file was created with 
        # a different version of pandas where pandas.core.indexes.numeric existed, but this module does not exist in your current pandas environment. 
        # This could be due to a major version difference or change in internal structure of pandas across versions.
    
    # print rhs data types
    
    # print(loaded_data['df_rhs'].dtypes)
    
    # send to csv to check data types
    
    # loaded_data['df_rhs'].to_csv('rhs.csv')

    # moodys_region_South                     float64
    # moodys_region_West                      float64
    # segment_Compact Crossover/SUV           float64
    # segment_Compact Luxury Crossover/SUV    float64
    # ... 
    # body_type_Regular Cab                   float64
    # body_type_Roadster                      float64
    # body_type_Sedan                         float64
    # body_type_Utility                       float64
    # body_type_Wagon                         float64
    
    # As I suspected... the data types are all floats. This is because the model was created using dummies instead of patsy C() syntax for categorical variables.
    # Meaning incoming data must be in the same format as the model was trained on.

    # add rhs variabls to api pickle object
    api['rhs'] = loaded_data['df_rhs']

    # Create fitted model
    loaded_model=sm.WLS(loaded_data['df_lhs'], loaded_data['df_rhs'], weights=loaded_data['weights'])
    
    fitted_model=loaded_model.fit(cov_type=loaded_data['cov_type'])
    
    # add model to api object
    api['model'] = fitted_model

    # load lookup dataset
    
    lookup_no_trim = pd.read_pickle(fr"O:\autocycle\AC_EV\lookup_tables\{ver}nadavin.bb_lookup_vin_notrim_qtile_mcm.pkl") #stata api incubator uses qtile aka qtile mcm which holds model_calib
    
    # Convert model_year to int
    
    lookup_no_trim['model_year'] = lookup_no_trim['model_year'].astype(int) # for method_loader.py, not used in forecasting, int_age_year_spline is used, int_age = model_year
    
    # Convert transmission empty strings to 'N/A

    lookup_no_trim['transmission'] = lookup_no_trim['transmission'].replace(' ', 'N/A')   
    
    # Create make_main and segment_main variables as copy of make and segment
    
    lookup_no_trim['make_main'] = lookup_no_trim['make']
    
    # Capitalize make_main
    
    lookup_no_trim['make_main'] = lookup_no_trim['make_main'].str.upper()
    
    # Create segment_main variable as copy of segment
    
    lookup_no_trim['segment_main'] = lookup_no_trim['segment']
    
    # Apply mappings to lookup_no_trim for make_main and segment_main, this is to create interaction terms for the model later on. 
    # Categorical variables have to be converted to dummies to interact with continuous variables in the model like mage, etc.
    
    # Load the mappings DataFrame from CSV file
    cat_mappings_df = pd.read_csv(fr"O:\autocycle\AC_EV\model_dev\2023\dev_output\reestimate_forecast\model_55\{ver}make_segment_mappings.csv")
    
    # Convert DataFrame back into a dictionary 
    cat_mappings = {row['Variable']:eval(row['Mapping']) for index, row in cat_mappings_df.iterrows()}
    
    # Create a list of float variables
    
    float_vars = ['segment_main', 'make_main']
    
    for var in float_vars:
        lookup_no_trim[var] = lookup_no_trim[var].map(cat_mappings[var])
    
    # print(lookup_no_trim.dtypes)
    
    # vin               object
    # icar              object
    # model_year        int32
    # msrp              float32
    # med_msrp          float32
    # make               object
    # model              object
    # segment            object
    # fuel_type          object
    # liters             object
    # cylinders          object
    # drive_type         object
    # body_type          object
    # induction_type     object
    # transmission       object
    # truck               int64
    # car                 int64
    # luxury              int64
    # iseg               object
    # trim_level        float32
    # dtype: object
    
    lookup_no_trim.rename(columns = {'vin':'short_vin', 'liters':'mod_liters', 'cylinders':'mod_cylinders',
                        'drive_type':'mod_drive_type', 'body_type':'mod_body_type', 'segment': 'mod_segment',
                        'fuel_type':'mod_fuel_type','induction_type':'mod_induction_type', 'make':'mod_make',
                        'msrp':'mod_msrp', 'trim_level':'mod_trim_level', 'model_year':'mod_model_year'}, inplace=True)
    # wherever model_calib is nan, replace with 0
    
    lookup_no_trim['model_calib'] = lookup_no_trim['model_calib'].replace(np.nan, 0)
    
    # same with vqi_adj
    
    lookup_no_trim['vqi_adj'] = lookup_no_trim['vqi_adj'].replace(np.nan, 0)

    # add lookup to api object
    api['lookup_no_trim'] = lookup_no_trim
    
    pd.set_option('future.no_silent_downcasting', True)
    
    for scen in scenlist:

        # Import tecon data that also has the vintage variables (VECON)
        
        Full_econ = pd.read_pickle(fr'O:\autocycle\AC_EV\econ_data\econ\{ver}nadavin.in_Fecon_'+ f'{scen}.pkl') 
        
        # Convert mtimes to datetime YM format
    
        Full_econ['mtime'] = pd.to_datetime(Full_econ['mtime'], format='%Y-%m')
        
        # All the txx and vxx variables are in the Full_econ dataframe, split txx into tecon data and vxx into vecon data
        
        tecon = Full_econ.filter(regex='^txx_')
        
        # Generating auto-related econ variables
        tecon['txx_stockpop'] = tecon['txx_fregfhav']/tecon['txx_fpop16gq']
        tecon['txx_gasdeflate'] = tecon['txx_fcpiuetb01']/tecon['txx_fcpiu']
        
        # keep only columns needed for forecasting
        tecon['mtime'] = Full_econ['mtime']
        tecon['txxtdlog_fscard'] = Full_econ['txxtdlog_fscard']
        tecon['txxyp_frprime'] = Full_econ['txxyp_frprime']
        # tecon['txxtdlog_fvhirneleaq'] = Full_econ['txxtdlog_fvhirneleaq']
        
        
        tecon = tecon[['mtime', 'txx_stockpop', 'txx_gasdeflate', 'txx_fvhirncaq', 
                    'txx_fvhirntaq', 'txx_fvhirneleaq', 'txxyp_frprime', 'txx_fip335', 'txx_fscard',
                    'txxtdlog_fscard','txx_fcpiu', 'txx_fcpiuehf', 'txx_fypdpiq', 'txx_flbr']] # 'txxtdlog_fvhirneleaq'

        for col in tecon.columns:
            if col!='mtime':
                # Replace 'ND' with NaN and immediately drop NaN values without creating an intermediate copy
                tecon[col] = tecon[col].replace('ND', np.nan).dropna()
                tecon[col] = tecon[col].infer_objects(copy=False)
                tecon[col] = tecon[col].astype(float)
        
        # Sort by mtime
        
        tecon = tecon.sort_values(by='mtime')
        
        # reset index
        
        tecon = tecon.reset_index(drop=True)
        
        #  Send to CSV for qc
        
        # tecon.to_csv('tecon.csv') # 2042-12-01
        
        # add tecon scen dataframe to api object    
        
        api[f'tecon_{scen}'] = tecon
        
        # Keep only vxx variables in vecon dataframe
        
        vecon = Full_econ.filter(regex='^vxx_')
        
        # Generating auto-related econ variables
        vecon['vxx_carpop'] = vecon['vxx_frcar']/vecon['vxx_fpop16gq']
        vecon['vxx_tcklpop'] = vecon['vxx_frtckl']/vecon['vxx_fpop16gq']
        vecon['vxx_gasdeflate'] = vecon['vxx_fcpiuetb01']/vecon['vxx_fcpiu']
        
        # keep only columns needed for forecastingabs
        vecon['mtime'] = Full_econ['mtime']
        vecon['vxxyp_fvhirncaq'] = Full_econ['vxxyp_fvhirncaq']
        vecon['vxxtdlog_fscard'] = Full_econ['vxxtdlog_fscard']
        # vecon['vxxtdlog_fvhirneleaq'] = Full_econ['vxxtdlog_fvhirneleaq']
        
        vecon = vecon[['mtime', 'vxx_carpop', 'vxx_tcklpop', 'vxx_gasdeflate', 'vxx_fvhirncaq', 'vxx_fvhirntaq', 'vxxyp_fvhirncaq', 'vxx_fvhirneleaq',
                    'vxxtdlog_fscard']] # 'vxxtdlog_fvhirneleaq'
        
        vecon['vxx_fvhirntaq'] = vecon['vxx_fvhirntaq'].replace('ND', np.nan).dropna()
        vecon['vxx_fvhirntaq']= vecon['vxx_fvhirntaq'].infer_objects(copy=False)
        vecon['vxx_fvhirntaq'] = vecon['vxx_fvhirntaq'].astype(float)

        vecon['vxx_fvhirncaq'] = vecon['vxx_fvhirncaq'].replace('ND', np.nan).dropna()
        vecon['vxx_fvhirncaq'] = vecon['vxx_fvhirncaq'].infer_objects(copy=False)
        vecon['vxx_fvhirncaq'] = vecon['vxx_fvhirncaq'].astype(float)
        # vecon['vxx_fvhirneleaq'] = vecon['vxx_fvhirneleaq'].replace('ND', np.nan).dropna().astype(float)

        for col in vecon.columns:
            if col!='mtime':
                # Replace 'ND' with NaN and immediately drop NaN values without creating an intermediate copy
                vecon[col] = vecon[col].replace('ND', np.nan).dropna()
                vecon[col] = vecon[col].infer_objects(copy=False)
                vecon[col] = vecon[col].astype(float)
        
        # Sort by mtime
        
        vecon = vecon.sort_values(by='mtime')
        
        # reset index
        
        vecon = vecon.reset_index(drop=True)
        
        # add vecon scen data to api object
        
        api[f'vecon_{scen}'] = vecon
        
        # print(api.keys()) # dict_keys(['rhs', 'model', 'lookup_no_trim', 'tecon_bl', 'vecon_bl'])
        
        #archive old api pickle file to "O:\autocycle\AC_EV\API\ACEV_API\Archived_APIs"
        
        if os.path.exists(fr'lambda-ev-forecast\app\{last_ver}api.pickle'):
            shutil.move(fr'lambda-ev-forecast\app\{last_ver}api.pickle', fr'O:\autocycle\AC_EV\API\Archived_APIs\{last_ver}api.pickle')
            # print(api.keys()) # dict_keys(['rhs', 'model', 'lookup_no_trim', 'tecon_bl', 'vecon_bl'])
        
        # save new api object to pickle file
        
        with open(fr'lambda-ev-forecast\app\{ver}api.pickle',"wb") as fp: # cd to the directory where the api.pickle file will be located
            
            pickle.dump(api,fp)
        
        # Add the pickle file to git ignore
        
        with open(fr'lambda-ev-forecast\app\.gitignore', 'a') as f:
            f.write(f'{ver}api.pickle\n')
            
        # load method_loader.py file
            
        with open(fr'lambda-ev-forecast\app\method_loader.py') as fp: # cd to the directory where the method_loader.py file is located
            
            methods['loader'] = compile(fp.read(),"method_loader",'exec') 
        
    print('API and methods objects created successfully')

    return [api,methods]

# ver is 'v' + this year + this month

ver = 'v' + datetime.datetime.now().strftime('%Y%m')

# last ver is 'v' + this year + last month

last_ver = 'v' + datetime.datetime.now().strftime('%Y') + str(int(datetime.datetime.now().strftime('%m'))-1).zfill(2)

# Creates the api and methods objects for the inperf API.
api,methods = create_inperf(ver,['bl','s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'])

# scenario list for inperf API

# scenlist = ['bl', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

########################################################################



# current_month = datetime.datetime.now().month

# if current_month < 10:
#     current_month = f'0{current_month}'
# else:
#     current_month = f'{current_month}'
    
# current_year = datetime.datetime.now().year

# mdl_num = '55' # this will have to be changed when model gets updated

# ver = 'v202408' # this will have to be changed when model gets updated

# scenlist = ['bl', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'] # this will have to be changed when model gets updated

#print(methods) {'get': {}, 'post': {}, 'loader': <code object <module> at 0x000001BF049D3030, file "method_loader", line 1>}

# # Load the dictionary from the pickle file
# df_dict = pd.read_pickle(r'D:\Users\garciac1\lambda-ev-forecast\ACEV-AWS-Lambda\lambda-ev-forecast\autocycle-ev-forecast\app\api.pickle')

# # Inspect the dictionary
# print("Dictionary Keys:", df_dict.keys())
# for key, value in df_dict.items():
#     print(f"Key: {key}, Type: {type(value)}")

# print(api) 
# 'model': <statsmodels.regression.linear_model.RegressionResultsWrapper object at 0x000001B43121C310>
# 'lookup_no_trim': 0x000001B43121C310

# print(methods) 
# {'get': {}, 'post': {}, 'loader': <code object <module> at 0x000001B4BC12D190, file "method_loader", line 1>}


