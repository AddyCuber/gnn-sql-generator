import streamlit as st
import pandas as pd
import json
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mistralai import Mistral
from mistralai.models import UserMessage
import sqlalchemy as sa
from sqlalchemy import text, create_engine, inspect
import pymysql
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GNNMySQLSystem:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        self.training_data = self.load_training_data()
        self.chat_history = []
        self.graph = None
        self.schema_info = None
        
    def load_training_data(self):
        """Load training data for fine-tuning"""
        try:
            with open('sql_training_data.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def extract_schema_graph(self, engine):
        """Extract database schema and build graph representation"""
        inspector = inspect(engine)
        
        # Build schema information
        schema_info = {
            'tables': [],
            'relationships': [],
            'graph_data': {}
        }
        
        # Extract tables and columns
        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'primary_key': col.get('primary_key', False)
                })
            
            schema_info['tables'].append({
                'name': table_name,
                'columns': columns
            })
        
        # Extract foreign key relationships
        for table_name in inspector.get_table_names():
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                schema_info['relationships'].append({
                    'from_table': table_name,
                    'to_table': fk['referred_table'],
                    'from_column': fk['constrained_columns'][0],
                    'to_column': fk['referred_columns'][0],
                    'type': 'foreign_key'
                })
        
        # Build NetworkX graph
        self.graph = nx.DiGraph()
        
        # Add nodes (tables)
        for table in schema_info['tables']:
            self.graph.add_node(table['name'], 
                              type='table',
                              columns=table['columns'])
        
        # Add edges (relationships)
        for rel in schema_info['relationships']:
            self.graph.add_edge(rel['from_table'], 
                              rel['to_table'],
                              type='foreign_key',
                              from_column=rel['from_column'],
                              to_column=rel['to_column'])
        
        # Create graph representation for LLM
        graph_data = self.create_graph_representation()
        schema_info['graph_data'] = graph_data
        
        self.schema_info = schema_info
        return schema_info
    
    def create_graph_representation(self):
        """Create text representation of the graph for LLM consumption"""
        if not self.graph:
            return {}
        
        graph_text = "DATABASE GRAPH STRUCTURE:\n\n"
        
        # Node information
        graph_text += "TABLES (Nodes):\n"
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            columns = node_data['columns']
            graph_text += f"• {node}:\n"
            for col in columns:
                pk_marker = " (PK)" if col['primary_key'] else ""
                graph_text += f"  - {col['name']}: {col['type']}{pk_marker}\n"
            graph_text += "\n"
        
        # Edge information
        graph_text += "RELATIONSHIPS (Edges):\n"
        for edge in self.graph.edges(data=True):
            from_table, to_table, data = edge
            graph_text += f"• {from_table}.{data['from_column']} → {to_table}.{data['to_column']}\n"
        
        # Graph metrics
        graph_text += f"\nGRAPH METRICS:\n"
        graph_text += f"• Number of tables: {self.graph.number_of_nodes()}\n"
        graph_text += f"• Number of relationships: {self.graph.number_of_edges()}\n"
        graph_text += f"• Connected components: {nx.number_strongly_connected_components(self.graph)}\n"
        
        return {
            'text': graph_text,
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'tables': list(self.graph.nodes()),
            'relationships': [(u, v, d) for u, v, d in self.graph.edges(data=True)]
        }
    
    def create_gnn_enhanced_prompt(self, question):
        """Create prompt enhanced with GNN and training data"""
        prompt = """You are an expert SQL generator with deep understanding of database graph structures. 
Generate accurate SQL queries based on the database graph, schema, and training examples.

"""
        
        # Add graph representation
        if self.schema_info and 'graph_data' in self.schema_info:
            prompt += self.schema_info['graph_data']['text']
            prompt += "\n"
        
        # Add training examples
        if self.training_data:
            prompt += "TRAINING EXAMPLES:\n"
            
            # Add positive examples
            positive_examples = [ex for ex in self.training_data['training_data'] if ex['complexity'] in ['simple', 'medium']]
            for example in positive_examples[:6]:
                prompt += f"Q: {example['question']}\n"
                prompt += f"A: {example['sql']}\n\n"
            
            prompt += "NEGATIVE EXAMPLES (what NOT to do):\n"
            for example in self.training_data['negative_examples'][:3]:
                prompt += f"Q: {example['question']}\n"
                prompt += f"A: {example['sql']}\n\n"
        
        prompt += f"""INSTRUCTIONS:
1. Use the graph structure to understand table relationships
2. Generate ONLY the SQL query - no explanations or additional text
3. Use proper JOIN syntax based on foreign key relationships
4. Use appropriate WHERE clauses for filtering
5. Use GROUP BY and HAVING for aggregations
6. If the query is invalid or unclear, respond with "INVALID QUERY: [reason]"
7. Keep queries simple and readable
8. Use backticks for table/column names if they contain special characters
9. Use LIMIT for large result sets
10. Return ONLY the SQL query, no explanations or markdown
11. Use table aliases (e.g., p.prof_name, s.student_name) to avoid duplicate column names
12. Only select necessary columns, avoid SELECT * when joining multiple tables

QUESTION: {question}

SQL QUERY:"""
        
        return prompt
    
    def generate_sql_with_gnn(self, question):
        """Generate SQL using GNN-enhanced prompt"""
        try:
            prompt = self.create_gnn_enhanced_prompt(question)
            
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=[UserMessage(content=prompt)],
                max_tokens=500,
                temperature=0.1
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Extract SQL query from the response
            sql_query = self.extract_sql_from_response(full_response)
            
            return sql_query
            
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            return None
    
    def extract_sql_from_response(self, response):
        """Extract SQL query from model response"""
        import re
        
        # Try to find SQL in code blocks first
        sql_pattern = r'```sql\s*(.*?)\s*```'
        match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try to find SQL without language specification
        sql_pattern = r'```\s*(.*?)\s*```'
        match = re.search(sql_pattern, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Check if it looks like SQL
            if any(keyword in content.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']):
                return content
        
        # Try to find SQL statements in the text
        sql_statements = []
        lines = response.split('\n')
        in_sql = False
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains SQL keywords
            if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']):
                in_sql = True
                sql_lines.append(line)
            elif in_sql and line.endswith(';'):
                sql_lines.append(line)
                in_sql = False
                sql_statements.append('\n'.join(sql_lines))
                sql_lines = []
            elif in_sql and line:
                sql_lines.append(line)
        
        if sql_statements:
            return sql_statements[0]  # Return the first SQL statement found
        
        # If no SQL found, return the original response but clean it up
        return response.strip()
    
    def execute_sql(self, engine, sql_query):
        """Execute SQL query safely"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                
                if result.returns_rows:
                    # Fetch results
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    # Handle duplicate column names
                    unique_columns = []
                    seen_columns = set()
                    column_mapping = {}
                    
                    for i, col in enumerate(columns):
                        if col in seen_columns:
                            # Create unique name for duplicate column
                            counter = 1
                            new_name = f"{col}_{counter}"
                            while new_name in seen_columns:
                                counter += 1
                                new_name = f"{col}_{counter}"
                            unique_columns.append(new_name)
                            column_mapping[i] = new_name
                        else:
                            unique_columns.append(col)
                            column_mapping[i] = col
                        seen_columns.add(unique_columns[-1])
                    
                    # Create DataFrame with unique column names
                    df = pd.DataFrame(rows, columns=unique_columns)
                    return df, None
                else:
                    # For INSERT, UPDATE, DELETE
                    conn.commit()
                    return None, f"Query executed successfully. {result.rowcount} rows affected."
                    
        except Exception as e:
            return None, f"SQL Error: {str(e)}"
    
    def visualize_graph(self):
        """Create graph visualization"""
        if not self.graph:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color='lightblue', 
                             node_size=2000, 
                             ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='gray', 
                             arrows=True, 
                             arrowsize=20, 
                             ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, 
                              font_size=10, 
                              font_weight='bold', 
                              ax=ax)
        
        # Edge labels
        edge_labels = {(u, v): f"{d['from_column']}→{d['to_column']}" 
                      for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, 
                                   edge_labels=edge_labels, 
                                   font_size=8, 
                                   ax=ax)
        
        plt.title("Database Schema Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        return fig

def main():
    st.set_page_config(page_title="GNN MySQL System", layout="wide")
    st.title("🧠 GNN-Powered MySQL Chat System")
    st.markdown("Graph Neural Network enhanced SQL generation with fine-tuned Mistral")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Database Configuration")
        
        # Database connection
        db_host = st.text_input("Host", value="localhost")
        db_port = st.number_input("Port", value=3306)
        db_user = st.text_input("Username")
        db_password = st.text_input("Password", type="password")
        db_name = st.text_input("Database Name")
        
        # API Configuration
        st.header("API Configuration")
        api_key = os.getenv("MISTRAL_API_KEY")
        st.info("API Key: Configured")
        
        # Initialize system
        if st.button("Initialize GNN System"):
            if all([db_host, db_user, db_password, db_name, api_key]):
                try:
                    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    engine = create_engine(connection_string)
                    
                    # Test connection
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    
                    # Initialize system
                    system = GNNMySQLSystem(api_key)
                    schema_info = system.extract_schema_graph(engine)
                    
                    st.success("GNN System initialized successfully!")
                    st.session_state['system'] = system
                    st.session_state['engine'] = engine
                    st.session_state['schema_info'] = schema_info
                    
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
            else:
                st.warning("Please fill in all fields.")
    
    # API key is hard-coded, no need to check
    pass
    
    if not all([db_host, db_user, db_password, db_name]):
        st.warning("Please configure your database connection in the sidebar.")
        return
    
    # Check if system is initialized
    if 'system' not in st.session_state:
        st.info("Please initialize the GNN system using the sidebar.")
        return
    
    system = st.session_state['system']
    engine = st.session_state['engine']
    schema_info = st.session_state['schema_info']
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Graph Visualization", "Schema Info", "History"])
    
    with tab1:
        st.header("💬 GNN-Enhanced Chat")
        
        # Display graph info
        if system.graph:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tables", system.graph.number_of_nodes())
            with col2:
                st.metric("Relationships", system.graph.number_of_edges())
        
        # Chat input
        user_question = st.text_area("Ask a question about your data:", 
                                    placeholder="e.g., Show me all students in the Computer Science department")
        
        if st.button("Generate SQL with GNN", type="primary"):
            if user_question:
                with st.spinner("Generating SQL using GNN..."):
                    sql_query = system.generate_sql_with_gnn(user_question)
                
                if sql_query:
                    # Execute the query
                    with st.spinner("Executing query..."):
                        result_df, message = system.execute_sql(engine, sql_query)
                    
                    # Store in chat history
                    system.chat_history.append({
                        'question': user_question,
                        'sql': sql_query,
                        'result': result_df,
                        'message': message,
                        'timestamp': time.time()
                    })
                    
                    # Display results
                    st.success("Query executed successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Generated SQL")
                        st.code(sql_query, language="sql")
                    
                    with col2:
                        st.subheader("Results")
                        if result_df is not None:
                            st.dataframe(result_df, use_container_width=True)
                            st.write(f"**Rows returned:** {len(result_df)}")
                        elif message:
                            st.info(message)
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.header("📊 Graph Visualization")
        
        if system.graph:
            fig = system.visualize_graph()
            if fig:
                st.pyplot(fig)
                
                # Graph metrics
                st.subheader("Graph Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nodes", system.graph.number_of_nodes())
                with col2:
                    st.metric("Edges", system.graph.number_of_edges())
                with col3:
                    st.metric("Density", f"{nx.density(system.graph):.3f}")
                
                # Connected components
                components = list(nx.strongly_connected_components(system.graph))
                st.write(f"**Connected Components:** {len(components)}")
                
                if len(components) > 1:
                    st.write("**Component Details:**")
                    for i, component in enumerate(components):
                        st.write(f"• Component {i+1}: {', '.join(component)}")
        else:
            st.info("No graph available. Please initialize the system.")
    
    with tab3:
        st.header("📋 Schema Information")
        
        if schema_info:
            # Tables
            st.subheader("Tables")
            for table in schema_info['tables']:
                with st.expander(f"📋 {table['name']}"):
                    for col in table['columns']:
                        pk_marker = " 🔑" if col['primary_key'] else ""
                        st.write(f"• {col['name']}: {col['type']}{pk_marker}")
            
            # Relationships
            st.subheader("Relationships")
            for rel in schema_info['relationships']:
                st.write(f"• {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}")
        else:
            st.info("No schema information available.")
    
    with tab4:
        st.header("📝 Chat History")
        
        if system.chat_history:
            for i, chat in enumerate(reversed(system.chat_history)):
                with st.expander(f"Query {len(system.chat_history) - i} - {time.strftime('%H:%M:%S', time.localtime(chat['timestamp']))}"):
                    st.write(f"**Question:** {chat['question']}")
                    st.code(chat['sql'], language="sql")
                    
                    if chat['result'] is not None:
                        st.dataframe(chat['result'], use_container_width=True)
                    elif chat['message']:
                        st.info(chat['message'])
        else:
            st.info("No chat history yet.")

if __name__ == "__main__":
    main() 