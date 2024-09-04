# Define custom function directory
ARG FUNCTION_DIR="/function"

#FROM python:3.12 as build-image
FROM python:3.8-slim-buster

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Create the directory and copy all files
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}

# Navigate to FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}

# Install the function's dependencies from requirements.txt and awslambdaric
RUN pip install \
    --target ${FUNCTION_DIR} \
    -r requirements.txt \
    awslambdaric

# Use a slim version of the base Python image to reduce the final image size
FROM python:3.8-slim

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Set the working directory to the function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies and application code from the build-image
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Set the runtime interface client as the default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Assuming your app.py contains a function named lambda_handler
# Adjust "app.lambda_handler" to match the actual module and method name
CMD [ "awslambda.lambda_handler" ]


# https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package