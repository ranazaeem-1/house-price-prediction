#!/bin/bash

echo "🚀 Starting House Price Prediction App deployment..."

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
docker stop house-price-app 2>/dev/null || true
docker rm house-price-app 2>/dev/null || true

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t house-price-app .

# Run container
echo "▶️ Starting container..."
docker run -d \
    --name house-price-app \
    -p 8501:8501 \
    --restart unless-stopped \
    house-price-app

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Check container status
if docker ps | grep -q house-price-app; then
    echo "✅ Container is running successfully!"
    
    # Get public IP (works on EC2)
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    echo "🌐 Access your app at: http://$PUBLIC_IP:8501"
    
    echo "📝 To train models (first time setup):"
    echo "   docker exec -it house-price-app python -c \"import subprocess; subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', 'house_price_automl_mlflow.ipynb'])\""
else
    echo "❌ Container failed to start. Checking logs..."
    docker logs house-price-app
    exit 1
fi

echo "🎉 Deployment completed!"
