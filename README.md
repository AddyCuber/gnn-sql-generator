# BLT GNN SQL System

A Graph Neural Network (GNN) powered natural language to SQL generation system using Mistral AI API and MySQL database connectivity.

## 🚀 Features

- **Graph Neural Network (GNN) Enhancement**: Uses lightweight GNN for intelligent table relationship understanding and optimal JOIN path selection
- **Privacy-First Architecture**: Schema anonymization, PII detection, and query validation ensure data privacy
- **Graph-based Schema Understanding**: Extracts database schema and creates graph representations using NetworkX
- **Natural Language to SQL**: Converts natural language queries to SQL using Mistral AI API with GNN insights
- **Few-shot Learning**: Uses 50 training examples to improve SQL generation accuracy
- **Interactive Web UI**: Streamlit-based interface with chat, graph visualization, query history, and privacy dashboard
- **Database Connectivity**: Supports MySQL and SQLite databases
- **Audit & Compliance**: Complete query logging and privacy compliance reporting
- **Smart Query Optimization**: GNN-powered table relevance scoring and JOIN path optimization

## 📋 Requirements

- Python 3.8+
- MySQL database (or SQLite for testing)
- Mistral AI API key

## 🛠️ Installation

### Quick Install (Recommended)
```bash
git clone <your-repo-url>
cd BLT_GNN_SQL
./install.sh
```

### Manual Install
```bash
git clone <your-repo-url>
cd BLT_GNN_SQL
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Mistral API key
```

## 🔧 Configuration

1. **Environment Setup**: Copy `.env.example` to `.env` and add your Mistral API key
2. **Database Connection**: Configure database credentials in the Streamlit sidebar
3. **Training Data**: Customize `sql_training_data.json` with your own examples (optional)

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run gnn_mysql_system.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
BLT_GNN_SQL/
├── gnn_mysql_system.py          # Main application with GNN and privacy features
├── sql_training_data.json       # Training examples for few-shot learning
├── add_training_examples.py     # CLI tool for managing training examples
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 🎯 How It Works

1. **Schema Extraction**: Uses SQLAlchemy to extract database schema
2. **Graph Construction**: Creates NetworkX graph from foreign key relationships
3. **GNN Initialization**: Builds table embeddings and calculates relationship weights
4. **Privacy Layer**: Anonymizes schema for LLM consumption while preserving structure
5. **Smart Prompt Engineering**: Builds context-aware prompts with GNN insights and training examples
6. **SQL Generation**: Sends privacy-safe prompts to Mistral AI API for SQL generation
7. **Privacy Validation**: Validates generated queries against privacy rules
8. **Query Execution**: Executes validated SQL and displays results
9. **Audit Logging**: Records all queries for compliance and monitoring
10. **Graph Visualization**: Shows database schema as interactive graph

## 📊 Training Data

The system uses 50 training examples (25 positive, 25 negative) covering:
- Simple SELECT queries
- Complex JOIN operations
- Aggregation functions
- Subqueries
- Error handling examples

## 🔒 Privacy & Security

- **Schema Anonymization**: Real table/column names are anonymized before sending to LLM
- **PII Detection**: Automatically identifies and handles personally identifiable information
- **Query Validation**: Blocks dangerous patterns and enforces privacy rules
- **Audit Trail**: Complete logging of all queries for compliance
- **Access Controls**: User-based query tracking and monitoring
- **API Security**: API keys stored in `.env` file (not committed to git)
- **Database Security**: Database credentials should be configured securely
- **Zero Data Exposure**: No actual data values sent to external APIs, only schema structure

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

- [ ] Deep GNN implementation with PyTorch Geometric
- [ ] Model fine-tuning pipeline for domain-specific queries
- [ ] Support for more database types (PostgreSQL, Oracle, etc.)
- [ ] Advanced query optimization and cost analysis
- [ ] Multi-language support for international deployments
- [ ] Query performance analysis and recommendations
- [ ] Role-based access controls and permissions
- [ ] Real-time privacy monitoring and alerts
- [ ] Integration with data governance platforms 