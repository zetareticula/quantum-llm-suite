#!/bin/bash

# Setup env (example tokens; replace with real)
echo "IBM_TOKEN=your_ibm_token" > .env
echo "RIGETTI_TOKEN=your_rigetti_token" >> .env
echo "IONQ_TOKEN=your_ionq_token" >> .env
echo "AWS_ACCESS_KEY_ID=your_aws_access_key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your_aws_secret_key" >> .env
echo "AWS_DEFAULT_REGION=us-west-1" >> .env

# Install deps
pip install -r requirements.txt
npm install --prefix frontend

# Run tests
pytest tests/

# Prod backend with Gunicorn (4 workers)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000

# Note: For NGINX proxy, add config:
# sudo apt install nginx
# Edit /etc/nginx/sites-available/default:
# location / {
#     proxy_pass http://127.0.0.1:8000;
# }
# sudo nginx -s reload