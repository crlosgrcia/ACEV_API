import os
import pandas as pd
import numpy as np
from scipy.special import expit
import boto3
import logging
import pickle
import datetime
from datetime import date, timedelta
import time
import math
import re
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from typing import List


# Have docker running
# sam local invoke

##############################################  Load API

#s3 = boto3.client('s3')

# - Loads the API object from api.pickle.
#    - Initializes an empty methods dictionary.
#    - Reads and compiles the content of method_loader.py, storing the compiled code in methods['loader'].
#    - Returns the api object and the methods dictionary.

# def load_inperf(ver):
    # Load the API Object
    #doc = s3.get_object(Bucket='ma-nprd-atoms-api-repository', Key=f'ATOMS/ACEV/iusa/{ver}/api.pickle') # change when ready
    # api = pickle.loads(doc['Body'].read()) # load the API pickle - S3 style
    # Load the methods
    #doc = s3.get_object(Bucket='ma-nprd-atoms-api-repository', Key=f'ATOMS/ACEV/iusa/{ver}/method_loader.py') # change when ready
    # methods = {'get':{},'post':{}} # initialize the methods dictionary
    # methods['loader'] = compile(doc['Body'].read().decode(),'method_loader','exec') # compile the method_loader.py
    # designed to compile Python code obtained from a remote source, such as an object in an AWS S3 bucket. 
    # doc['Body'] suggests that doc is likely a response object from a request to a remote storage service, where Body is a stream representing the body of the response.
    # - **Reading Method**: The read() method is used to read the binary content of the response body. 
    # This is followed by decode(), which converts the binary data into a UTF-8 encoded string. This step is crucial because compile() expects a string of Python code.
    
    ##############################################  Local AWS lambda mock test Load API #####################################

def load_inperf(ver):
    with open(fr'lambda-ev-forecast\app\v202409api.pickle',"rb") as fp:
        api = pd.read_pickle(fp)
    with open(fr'lambda-ev-forecast\app\method_loader.py') as fp:
        methods = {'get':{},'post':{}}
        methods['loader'] = compile(fp.read(),"method_loader",'exec') 
    return [api,methods]


def convert_types(o):
    if isinstance(o,np.generic): return o.item()
    raise TypeError

def return_json(json): # constructs a dictionary that mimics the structure of an AWS API Gateway response object
    response = {
        "statusCode": 200,
        "statusDescription": "200 OK",
        "isBase64Encoded": False,
        "headers": {
            "Content-Type": "application/json; charset=utf-8"
        }
    }
    response['body'] = json
    return response

##############################################  UI ####################################

# def return_html():
#     response = {
#         "statusCode": 200,
#         "statusDescription": "200 OK",
#         "isBase64Encoded": False,
#         "headers": {
#             "Content-Type": "text/html; charset=utf-8"
#         }
#     }
#     with open('./GUI/index.html') as fp:
#         payload = fp.read()  
#     response['body'] = payload
#     return response

# def return_css():
#     response = {
#         "statusCode": 200,
#         "statusDescription": "200 OK",
#         "isBase64Encoded": False,
#         "headers": {
#             "Content-Type": "text/css; charset=utf-8"
#         }
#     }
#     with open('./GUI/swagger-ui.css') as fp:
#         payload = fp.read()  
#     response['body'] = payload
#     return response

# def return_js(fname):
#     response = {
#         "statusCode": 200,
#         "statusDescription": "200 OK",
#         "isBase64Encoded": False,
#         "headers": {
#             "Content-Type": "application/javascript; charset=utf-8"
#         }
#     }
#     with open(f'./GUI/swagger-ui-{fname}.js') as fp:
#         payload = fp.read()  
#     response['body'] = payload
#     return response

##############################################  Lambda Handler ##########################################

# Vehicle object structure - used as a data validation layer and to provide default values for missing fields before "fixvehicles" function

def get_current_month():
    curmonth = date.today().replace(day=1)
    return curmonth.strftime("%Y-%m-%d")

def get_end_month():
    end_date = (date.today() + timedelta(days=365*5)).replace(day=1)
    return end_date.strftime("%Y-%m-%d")

class VehicleConfig:
    protected_namespaces = ()

class Vehicle(BaseModel): 

    # VIN is the primary key for the vehicle
    VIN: str = "5YJ3E1E1_H" 
    initial_mileage: Optional[int]
    annual_mileage_assumption: int = 10000
    msrp: int = 60000
    model_year: int = Field(..., description="Year of the vehicle model", alias="vehicle_model_year")
    moodys_region: Optional[str] = "South"
    sale_type: Optional[str] = "Unknown"
    interior_color: Optional[str] = "."
    exterior_color: Optional[str] = "."
    start_month :Optional[str] = Field(default_factory=get_current_month)
    end_month :Optional[str] = Field(default_factory=get_end_month)

    class Config(VehicleConfig):
        pass

# This creates the acceptable input for the lambda_handler function

class ForecastRequest(BaseModel):
    scenario: str = "bl" # Default value for scenario
    vehicles: List[Vehicle] # List of Vehicle objects that go into the request body above ^


def lambda_handler(event, context):
    # creates object to log messages
    logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)
    
    # get aws request id and set it in the event
    request_id = context.aws_request_id
    
    event['RequestId'] = request_id
    
    # get the uri and method
    uri = event["path"]

    method = uri.split('/')[-1]
    
    # get the query string parameters
    query = event["queryStringParameters"]
    
    # get the http method
    http_method = event['httpMethod'].lower()
    
    # log the request
    logger.info(json.dumps({'RequestId': context.aws_request_id, 'uri': event["path"]}))
    # logstuff = {'RequestId':request_id,'uri':uri}
    # logger.info(json.dumps(logstuff))
    
    parts = uri.split('/')
    
    # if the uri is /pyatoms/echo, return the event, USE THIS ONLY FOR DEV TO SEE WHAT IS BEING RETURNED BY THE API
    if uri == '/pyatoms/echo':
        return return_json(json.dumps(event))
    
    ########################################## Where is this swagger json file created?
    # elif (parts[2] == "swagger" or parts[2] == "help") and parts[-1] == "index.html":        
    #     resp = return_html()
    #     # TODO: Make this generic
    #     urls = [{'url':'/pyatoms/ACML/iusa/swagger.v1.json','name':'AutoCycle ML IUSA - V1'}]
    #     resp['body'] = resp['body'].replace("---URLS---",json.dumps(urls))
    #     return resp
    # elif uri.endswith("/swagger.v1.json"):
    #     swagger = s3.get_object(Bucket='ma-nprd-atoms-api-repository', Key="ATOMS/ACML/iusa/swagger.v1.json")
    #     contents = swagger['Body'].read().decode()
    #     contents = contents.replace("/atoms/AC/v1/iaus","/pyatoms/AC/v1/iaus")
    #     return return_json(contents)
    # elif uri.endswith("/swagger-ui.css"):
    #     return return_css()
    # elif uri.endswith("/swagger-ui-bundle.js"):
    #     return return_js('bundle')
    # elif uri.endswith("/swagger-ui-standalone-preset.js"):
    #     return return_js('standalone-preset')
    ########################################## Health API
    
    elif uri.endswith("/health"):
        if http_method == "post":
            return return_json(json.dumps({'status':'Python ATOMS is Alive!'}))
        
    ########################################## Forecast API
    elif uri.endswith("/forecast"):
        
        ver = query['modelVersion']
        
        if http_method == "post":
            
            payload = json.loads(event["body"])
            
        else:
            
            payload = {}
        
        api,methods = load_inperf(ver)
        
        # print(methods) {'get': {}, 'post': {}, 'loader': <code object <module> at 0x000001FC7690A1F0, file "method_loader", line 1>}
        #print(api.keys()) # dict_keys(['rhs', 'model', 'lookup_no_trim', 'tecon_bl', 'vecon_bl'])
        
        exec(methods['loader']) 
        
        # print(methods) #{'get': {}, 'post': {'forecast': <function forecast at 0x000001FC338559E0>}, 'loader': <code object <module> at 0x000001FC7690A1F0, file "method_loader", line 1>}
        
        # print(api.keys()) # dict_keys(['rhs', 'model', 'lookup_no_trim', 'tecon_bl', 'vecon_bl'])
        
        # print(api)
        
        # print(method) # forecast
        
        # print(http_method) # post
        
        # print(methods[http_method][method]) # <function forecast at 0x000001FC338559E0>
        
        methods[http_method][method](api,query,payload)
        
        return return_json(json.dumps(payload,default=convert_types))
    


############################################ test functions ######################################

# def test_return_json():
    # # Arrange
    # input_json = '{"key": "value"}'
    
    # # Act
    # response = return_json(statusCode=200, body=input_json)
    
    # # Assert
    # assert response == {
    #     "statusCode": 200,
    #     "statusDescription": "200 OK",
    #     "isBase64Encoded": False,
    #     "headers": {
    #         "Content-Type": "application/json; charset=utf-8"
    #     },
    #     'body': input_json
    # }
    
    # print("Test passed successfully!")

# Call the test function:
# test_return_json()

from unittest.mock import MagicMock
import json
def test_lambda_handler():
    # Mocking event 
    # Note: This dictionary should mimic actual event structure that received by lambda when invoked through AWS    
    event = {
        "httpMethod": "POST",
        "path": "/forecast",
        "queryStringParameters": {"modelVersion":"v202409"},
        "body": json.dumps({
            'scenario':'bl',
            'vehicles':[{'VIN':'5YJ3E1E1_H','initial_mileage':10000,'annual_mileage_assumption':8000,'msrp':37250,'model_year':2017}]
            })
    }

    # Mocking context (Just a sample)
    class Context:
        def __init__(self):
            self.aws_request_id = 'mocked request id'
            self.get_remaining_time_in_millis = MagicMock(return_value=5000)  # You may not need this    

    context = Context()

    response = lambda_handler(event=event, context=context)

    print(response)
    
# Call the test function:
test_lambda_handler()