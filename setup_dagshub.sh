#!/bin/bash

echo "=========================================="
echo "🚀 DagsHub Setup for MLOps Project"
echo "=========================================="
echo ""

# # Check if .env exists
# if [ -f ".env" ]; then
#     echo "⚠️  .env file already exists!"
#     read -p "Do you want to overwrite it? (y/N): " confirm
#     if [[ $confirm != [yY] ]]; then
#         echo "Setup cancelled."
#         exit 0
#     fi
# fi

# # Get user inputs
# echo "📝 Please provide your DagsHub credentials:"
# echo ""
# read -p "DagsHub Username: " DAGSHUB_USERNAME
# read -sp "DagsHub Token: " DAGSHUB_TOKEN
# echo ""
# echo ""

DAGSHUB_REPO="ml-forecast-system"
DAGSHUB_USERNAME="ameytrainer"
DAGSHUB_TOKEN="efe797c44cfe234ddb245b1a7e4aa61750ce0cac"

# Create .env file
# cat > .env << EOF
# # DagsHub MLflow Configuration
# DAGSHUB_USERNAME=$DAGSHUB_USERNAME
# DAGSHUB_TOKEN=$DAGSHUB_TOKEN
# DAGSHUB_REPO=$DAGSHUB_REPO

# # MLflow Tracking (auto-configured)
# MLFLOW_TRACKING_URI=https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO.mlflow
# MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
# MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

# # DVC Remote (auto-configured)
# DVC_REMOTE_URL=https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO.dvc
# EOF

# echo "✅ .env file created successfully!"
# echo ""

# Configure DVC remote
echo "🔧 Configuring DVC remote..."
dvc remote remove origin 2>/dev/null
echo "1"
dvc remote add origin https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO.dvc
echo "2"
dvc remote default origin
echo "3"
dvc remote modify origin --local auth basic
echo "4"
dvc remote modify origin --local user $DAGSHUB_USERNAME
echo "5"
dvc remote modify origin --local password $DAGSHUB_TOKEN
echo "6"

echo "✅ DVC remote configured!"
echo ""

# Test connection
echo "🧪 Testing DagsHub connection..."
if dvc push 2>&1 | grep -q "Everything is up to date"; then
    echo "✅ DVC connection successful!"
else
    echo "⚠️  DVC connection test skipped (no data yet)"
fi

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add GitHub secrets:"
echo "   - DAGSHUB_USERNAME: $DAGSHUB_USERNAME"
echo "   - DAGSHUB_TOKEN: (your token)"
echo ""
echo "2. Start local services:"
echo "   Terminal 1: uvicorn app.backend:app --reload --port 5000"
echo "   Terminal 2: streamlit run app/dashboard.py"
echo ""
echo "3. Make a change and push:"
echo "   vim params.yaml  # Change max_depth"
echo "   git commit -am 'Experiment: tune hyperparameters'"
echo "   git push origin main"
echo ""
echo "4. Watch the magic! 🎉"
echo "   GitHub: https://github.com/YOUR_USERNAME/ml-forecast-system/actions"
echo "   DagsHub: https://dagshub.com/$DAGSHUB_USERNAME/$DAGSHUB_REPO"
echo ""