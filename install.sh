#!/bin/bash

# Enhanced GNN MySQL System - Installation Script

echo "🚀 Installing Enhanced GNN MySQL System..."
echo "=========================================="

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created - please edit it and add your Mistral API key"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "=========================================="
echo "✨ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Mistral API key"
echo "2. Run the system: streamlit run gnn_mysql_system.py"
echo ""
echo "Features available:"
echo "🧠 GNN-enhanced SQL generation"
echo "🔒 Privacy-preserving schema anonymization"
echo "📊 Query audit and compliance reporting"
echo "📈 Interactive graph visualization"