import pickle as pickle
import json
import datetime
import numpy as np
import pandas as pd
import os
from scipy.special import expit

# print(os.getcwd())

def load_inperf(ver):
    with open(fr'lambda-ev-forecast\app\{ver}api.pickle',"rb") as fp:
        api = pd.read_pickle(fp)
    with open(fr'lambda-ev-forecast\app\method_loader.py') as fp:
        methods = {'get':{},'post':{}}
        methods['loader'] = compile(fp.read(),"method_loader",'exec') 
    return [api,methods]

api,methods = load_inperf('v202411')

print(api) # print api dict

# print keys in api dict

# print(api.keys()) 

# dict_keys(['model', 'lookup_no_trim', 'tecon_bl', 'vecon_bl', 'tecon_s0', 'vecon_s0', 'tecon_s1', 'vecon_s1', 'tecon_s2', 'vecon_s2', 'tecon_s3', 'vecon_s3', 'tecon_s4', 'vecon_s4', 'tecon_s5', 'vecon_s5', 't
# econ_s6', 'vecon_s6', 'tecon_s7', 'vecon_s7', 'tecon_s8', 'vecon_s8'])

# print(methods) {'get': {}, 'post': {}, 'loader': <code object <module> at 0x00000243C544FDF0, file "method_loader", line 1>}

exec(methods['loader']) # methods['loader'] is a code object

# print(methods) {'get': {}, 'post': {'forecast': <function forecast at 0x000002449FD74860>}, 'loader': <code object <module> at 0x00000243C544FDF0, file "method_loader", line 1>}

http_method = "post"
route = "/forecast"
method = route.split('/')[-1]
query = {'modelVersion':'v202411'}
payload = {
    'scenario':'s8', # can only do one scenario at a time.'bl', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', etc. found in inperf. 'mcr_u2', 'mcr_nd', 'mcr_rf', 'mcr_ep', 'mcr_cp', 'mcr_lp, not yet included
    'vehicles':
        [{'VIN':'1FT6W1EV_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #Ford F150
        {'VIN':'1N4AZ1BV_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #Nissan Leaf
        {'VIN':'5YJ3E1E1_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #Tesla Model 3
        {'VIN':'KNDC34AA_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #Kia Niro
        {'VIN':'7SAYCAED_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #Tesla Model Y
        {'VIN':'WBY53EJ0_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #BMW i7
        {'VIN':'WBY33AW0_P','initial_mileage':10000,'annual_mileage_assumption': 12000,'msrp':37250,'model_year':2023}, #BMW i4
        ] # Model Year doesn't matter because it gets updated by NoTrimLookup merge
    }

# columns in payload are vin, intial_mileage, annual_mileage_assumption, msrp, model_year, moodys_region, interior_color, sale_type, start_month, end_month,  exterior_color.
# Examples to Run from No Trim LookUp
# vin	            icar	      model_year    msrp	med_msrp	make	model	      segment	    fuel_type	liters	cylinders	drive_type	body_type	induction_type	transmission	truck	car	luxury	iseg	            trim_level: 
# 5YJ3E1E1_H	TESLA-Model 3-2017	2017	   35000	37250	   TESLA	Model 3	   Near Luxury Car	Electric	.	       .	      RWD	     Sedan	      Standard		                   0	1	1	     Near Luxury Car	0.9395973
# 5YJ3E1E1_P	TESLA-Model 3-2023	2023	   40240	48490	   TESLA	Model 3	   Near Luxury Car	Electric	.	        .	      RWD	     Sedan	      Standard	         A	           0	1	1	     Near Luxury Car	0.8298618
# 19XFA152_9	HONDA-Civic-2009	2009	   16305	18855	   HONDA	Civic	   Compact Car	      CNG	    1	        4	      FWD	     Sedan	      Standard		                   0	1	0	     Compact Car	    0.86475736

ver = query['modelVersion']
# api = load_pickles(ver)
output = methods[http_method][method](api,query,payload)

# print(api.keys())

# print(methods)

# print(methods) {'get': {}, 'post': {'forecast': <function forecast at 0x000002449FD74860>}, 'loader': <code object <module> at 0x00000243C544FDF0, file "method_loader", line 1>}

# for further analysis
df = pd.DataFrame(output)
df.to_csv(fr'O:\autocycle\AC_EV\API\S8forecast_11.csv',index=False)

print(output) # works like a charm
print("finished")