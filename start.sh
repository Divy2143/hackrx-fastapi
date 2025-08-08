#!/bin/bash

# Exit if any command fails
set -e

# Run the FastAPI app with Uvicorn
uvicorn main:app --host 0.0.0.0 --port $PORT
