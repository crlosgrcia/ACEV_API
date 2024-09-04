import json
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator 
from pydantic import BaseModel, ConfigDict
from pydantic import Field
from typing import Optional
from typing import List, Optional
from awslambda import lambda_handler  # Import the Lambda handler function from awslambda.py

# This simulates the API Gateway event and context objects, aka the Lambda invocation of app.py

awslambda = FastAPI()

@awslambda.post("/pyatoms/echo")
async def echo(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}

    return JSONResponse(content=body)

@awslambda.post("/health")
async def health(request: Request):
    return JSONResponse(content={"status": "Python ATOMS is Alive!"})

@awslambda.post("/forecast")
async def test_forecast():
    query = {"modelVersion": "v202407"} # load_inperf ver
    payload = {
        "scenario": "bl",
        "vehicles": [
            {"VIN": "1GT12TE8_G", "initial_mileage": 34601, "annual_mileage_assumption": 8000, "msrp": 60000, "model_year": 2019},
            {"VIN": "5NMZT3FB_H", "annual_mileage_assumption": 10000, "msrp": 50000, "model_year": 2019},
            {"VIN": "NOSUCHVI_N", "annual_mileage_assumption": 10000, "msrp": 50000, "model_year": 2019},
            {"VIN": "3GB5YTE7_M", "initial_mileage": 34601, "annual_mileage_assumption": 12000, "msrp": 140009,
            "moodys_region": "South", "interior_color": ".", "sale_type": "As Is", "model_year": 2019},
            {"VIN": "3C63R3LJ_K", "initial_mileage": 34601, "annual_mileage_assumption": 8000, "msrp": 60000, "model_year": 2019, "exterior_color": "PERIWINKLE"},
            {"VIN": "4S3BE896_4", "annual_mileage_assumption": 10000, "msrp": 50000, "model_year": 2019},
            {"VIN": "1GB5KYCY5J123", "moodys_region": "South", "initial_mileage": 34601,
            "annual_mileage_assumption": 10000, "msrp": 50000, "sale_type": "As Is", "interior_color": ".",
            "exterior_color": ".", "start_month": "1-Jan-2020", "end_month": "1-Dec-2024", "model_year": 2019}
        ]
    }

    event = {
        "path": "/forecast",
        "httpMethod": "POST",
        "queryStringParameters": query,
        "body": json.dumps(payload)
    }

    # Mock context object
    class MockContext:
        def __init__(self, aws_request_id):
            self.aws_request_id = aws_request_id

    context = MockContext("dummy-id-for-testing")

    lambda_response = lambda_handler(event, context)

    try:
        body = json.loads(lambda_response.get("body", "{}"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error processing lambda response")

    status_code = lambda_response.get("statusCode", 200)

    return JSONResponse(content=body, status_code=status_code)

# cd D:\Users\garciac1\lambda-ev-forecast\ACEV-AWS-Lambda\lambda-ev-forecast\autocycle-ev-forecast\app\     in anaconda exe/powershell terminal
# uvicorn main:awslambda --reload is a command used to run a FastAPI application using Uvicorn, with the --reload flag to enable auto-reloading.
# http://127.0.0.1:8000/docs or redoc
# ctrl to c to close the server
