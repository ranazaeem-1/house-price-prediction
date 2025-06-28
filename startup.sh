#!/bin/bash

# Startup script for EC2 deployment
set -e

echo "Starting House Price Prediction App deployment..."

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ] || [ "$(head -c 3 /sys/hypervisor/uuid)" != "ec2" ]; then
    echo "Warning: This script is designed for EC2 instances"
fi

# Update system
echo "Updating system packages..."
if command -v yum &> /dev/null; then
    sudo yum update -y
elif command -v apt &> /dev/null; then
    sudo apt update && sudo apt upgrade -y
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    if command -v yum &> /dev/null; then
        # Amazon Linux 2
        sudo yum install -y docker
        sudo service docker start
        sudo usermod -a -G docker ec2-user
    elif command -v apt &> /dev/null; then
        # Ubuntu
        sudo apt install -y docker.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ubuntu
    fi
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create necessary directories
mkdir -p models mlruns

# Check if dataset exists
if [ ! -f "House Price Prediction Dataset.csv" ]; then
    echo "Error: Dataset file 'House Price Prediction Dataset.csv' not found!"
    echo "Please upload the dataset file to the current directory."
    exit 1
fi

# Build and run the application
echo "Building and starting the application..."
docker-compose down 2>/dev/null || true
docker-compose build
docker-compose up -d

# Wait for application to start
echo "Waiting for application to start..."
sleep 30

# Check if application is running
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "Access your app at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
else
    echo "❌ Application failed to start. Check logs:"
    docker-compose logs
    exit 1
fi

echo "Deployment completed successfully!"
