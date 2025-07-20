# BLT GNN SQL System

A Graph Neural Network (GNN) powered natural language to SQL generation system using Mistral AI API and MySQL database connectivity.

## 🚀 Features

- **Graph-based Schema Understanding**: Extracts database schema and creates graph representations using NetworkX
- **Natural Language to SQL**: Converts natural language queries to SQL using Mistral AI API
- **Few-shot Learning**: Uses 50 training examples to improve SQL generation accuracy
- **Interactive Web UI**: Streamlit-based interface with chat, graph visualization, and query history
- **Database Connectivity**: Supports MySQL and SQLite databases
- **Privacy-Preserving**: Graph-based approach without exposing raw schema data

## 📋 Requirements

- Python 3.8+
- MySQL database (or SQLite for testing)
- Mistral AI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd BLT_GNN_SQL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```

## 🔧 Configuration

1. **Database Connection**: Update the database connection parameters in `gnn_mysql_system.py`
2. **API Key**: Add your Mistral AI API key to the `.env` file
3. **Training Data**: Customize `sql_training_data.json` with your own examples

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run gnn_mysql_system.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
BLT_GNN_SQL/
├── gnn_mysql_system.py          # Main application
├── sql_training_data.json       # Training examples
├── add_training_examples.py     # CLI tool for managing examples
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (not in git)
└── README.md                   # This file
```

## 🎯 How It Works

1. **Schema Extraction**: Uses SQLAlchemy to extract database schema
2. **Graph Construction**: Creates NetworkX graph from foreign key relationships
3. **Prompt Engineering**: Builds context-aware prompts with training examples
4. **SQL Generation**: Sends prompts to Mistral AI API for SQL generation
5. **Query Execution**: Executes generated SQL and displays results
6. **Graph Visualization**: Shows database schema as interactive graph

## 📊 Training Data

The system uses 50 training examples (25 positive, 25 negative) covering:
- Simple SELECT queries
- Complex JOIN operations
- Aggregation functions
- Subqueries
- Error handling examples

## 🔒 Security

- API keys stored in `.env` file (not committed to git)
- Database credentials should be configured securely
- No raw schema data sent to external APIs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`
2. **API Key Error**: Ensure your Mistral API key is correctly set in `.env`
3. **Database Connection**: Verify database credentials and connectivity
4. **Duplicate Columns**: The system automatically handles duplicate column names

### Getting Help

- Check the error logs in the Streamlit interface
- Verify your database connection parameters
- Ensure all dependencies are installed correctly

## 🔮 Future Enhancements

- [ ] True GNN implementation with neural network layers
- [ ] Model fine-tuning pipeline
- [ ] Support for more database types
- [ ] Advanced query optimization
- [ ] Multi-language support
- [ ] Query performance analysis 