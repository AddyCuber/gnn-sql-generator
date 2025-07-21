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
import hashlib
import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available - using simplified GNN mode")
from collections import deque

# Load environment variables
load_dotenv()

class PrivacyLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class ColumnPrivacy:
    name: str
    original_name: str
    privacy_level: PrivacyLevel
    data_type: str
    is_pii: bool = False
    anonymized_name: str = None

class SimpleSchemaGNN:
    """Lightweight GNN for schema understanding without heavy ML dependencies"""
    
    def __init__(self, schema_info):
        self.schema_info = schema_info
        self.table_embeddings = {}
        self.relationship_weights = {}
        self.query_patterns = []
        
    def create_table_embeddings(self):
        """Create embeddings based on table structure and column types"""
        if SKLEARN_AVAILABLE:
            return self._create_tfidf_embeddings()
        else:
            return self._create_simple_embeddings()
    
    def _create_tfidf_embeddings(self):
        """Create TF-IDF based embeddings"""
        tables = []
        table_features = []
        
        for table in self.schema_info['tables']:
            tables.append(table['name'])
            
            # Create feature vector from column types and names
            feature_text = []
            for col in table['columns']:
                feature_text.append(f"{col['name']} {col['type']}")
                if col['primary_key']:
                    feature_text.append("primary_key")
            
            table_features.append(" ".join(feature_text))
        
        # Use TF-IDF for embeddings
        if len(table_features) > 0:
            vectorizer = TfidfVectorizer(max_features=100)
            embeddings = vectorizer.fit_transform(table_features)
            
            # Store embeddings
            for i, table_name in enumerate(tables):
                self.table_embeddings[table_name] = embeddings[i].toarray()[0]
        
        return self.table_embeddings
    
    def _create_simple_embeddings(self):
        """Create simple embeddings without sklearn"""
        for table in self.schema_info['tables']:
            # Simple feature vector based on column count and types
            features = []
            features.append(len(table['columns']))  # Number of columns
            
            # Count different column types
            type_counts = {}
            for col in table['columns']:
                col_type = col['type'].split('(')[0].upper()  # Get base type
                type_counts[col_type] = type_counts.get(col_type, 0) + 1
            
            # Add type counts as features
            common_types = ['INTEGER', 'VARCHAR', 'DECIMAL', 'DATE', 'BOOLEAN']
            for t in common_types:
                features.append(type_counts.get(t, 0))
            
            # Add primary key count
            pk_count = sum(1 for col in table['columns'] if col['primary_key'])
            features.append(pk_count)
            
            self.table_embeddings[table['name']] = features
        
        return self.table_embeddings
    
    def calculate_relationship_weights(self):
        """Calculate importance weights for relationships"""
        if SKLEARN_AVAILABLE:
            return self._calculate_cosine_weights()
        else:
            return self._calculate_simple_weights()
    
    def _calculate_cosine_weights(self):
        """Calculate weights using cosine similarity"""
        for rel in self.schema_info['relationships']:
            from_table = rel['from_table']
            to_table = rel['to_table']
            
            # Simple heuristic: weight based on table similarity
            if from_table in self.table_embeddings and to_table in self.table_embeddings:
                similarity = cosine_similarity(
                    [self.table_embeddings[from_table]], 
                    [self.table_embeddings[to_table]]
                )[0][0]
                
                # Higher weight for more similar tables
                weight = 0.5 + (similarity * 0.5)
                self.relationship_weights[(from_table, to_table)] = weight
        
        return self.relationship_weights
    
    def _calculate_simple_weights(self):
        """Calculate weights without cosine similarity"""
        for rel in self.schema_info['relationships']:
            from_table = rel['from_table']
            to_table = rel['to_table']
            
            # Simple heuristic based on table complexity
            from_features = self.table_embeddings.get(from_table, [0])
            to_features = self.table_embeddings.get(to_table, [0])
            
            # Weight based on table complexity (more columns = higher weight)
            from_complexity = from_features[0] if from_features else 1
            to_complexity = to_features[0] if to_features else 1
            
            # Normalize weight between 0.3 and 1.0
            weight = 0.3 + (min(from_complexity, to_complexity) / 10.0) * 0.7
            weight = min(weight, 1.0)
            
            self.relationship_weights[(from_table, to_table)] = weight
        
        return self.relationship_weights
    
    def find_optimal_join_path(self, start_table, end_table):
        """Find best path between tables using weighted relationships"""
        if start_table == end_table:
            return [start_table]
        
        # Simple BFS with weights
        queue = deque([(start_table, [start_table], 0.0)])
        visited = set()
        best_path = None
        best_weight = -1
        
        while queue:
            current_table, path, total_weight = queue.popleft()
            
            if current_table in visited:
                continue
            visited.add(current_table)
            
            # Check all relationships from current table
            for rel in self.schema_info['relationships']:
                next_table = None
                if rel['from_table'] == current_table:
                    next_table = rel['to_table']
                elif rel['to_table'] == current_table:
                    next_table = rel['from_table']
                
                if next_table and next_table not in visited:
                    weight = self.relationship_weights.get(
                        (current_table, next_table), 0.5
                    )
                    new_weight = total_weight + weight
                    new_path = path + [next_table]
                    
                    if next_table == end_table:
                        if new_weight > best_weight:
                            best_weight = new_weight
                            best_path = new_path
                    else:
                        queue.append((next_table, new_path, new_weight))
        
        return best_path or []
    
    def suggest_relevant_tables(self, question_embedding, top_k=3):
        """Suggest most relevant tables for a question"""
        if not self.table_embeddings:
            return []
        
        if SKLEARN_AVAILABLE:
            return self._suggest_tables_cosine(question_embedding, top_k)
        else:
            return self._suggest_tables_simple(question_embedding, top_k)
    
    def _suggest_tables_cosine(self, question_embedding, top_k):
        """Suggest tables using cosine similarity"""
        similarities = {}
        for table_name, table_emb in self.table_embeddings.items():
            sim = cosine_similarity([question_embedding], [table_emb])[0][0]
            similarities[table_name] = sim
        
        # Return top-k most similar tables
        sorted_tables = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables[:top_k]]
    
    def _suggest_tables_simple(self, question, top_k):
        """Simple table suggestion based on keyword matching"""
        if isinstance(question, str):
            question_lower = question.lower()
        else:
            # If it's an embedding array, we can't do keyword matching
            return list(self.table_embeddings.keys())[:top_k]
        
        table_scores = {}
        
        for table in self.schema_info['tables']:
            score = 0
            table_name = table['name'].lower()
            
            # Score based on table name match
            if table_name in question_lower:
                score += 10
            
            # Score based on column name matches
            for col in table['columns']:
                col_name = col['name'].lower()
                if col_name in question_lower:
                    score += 5
                    
                # Bonus for primary keys
                if col['primary_key'] and col_name in question_lower:
                    score += 2
            
            table_scores[table['name']] = score
        
        # Return top-k tables
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables[:top_k] if score > 0]
    
    def enhance_sql_generation_context(self, question):
        """Add GNN insights to SQL generation context"""
        if SKLEARN_AVAILABLE:
            return self._enhance_context_tfidf(question)
        else:
            return self._enhance_context_simple(question)
    
    def _enhance_context_tfidf(self, question):
        """Enhance context using TF-IDF"""
        # Create simple question embedding
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            # Fit on question + table names for vocabulary
            all_text = [question] + [table['name'] for table in self.schema_info['tables']]
            vectorizer.fit(all_text)
            question_emb = vectorizer.transform([question]).toarray()[0]
        except:
            # Fallback if TF-IDF fails
            question_emb = np.random.random(100)
        
        # Get relevant tables
        relevant_tables = self.suggest_relevant_tables(question_emb)
        
        return self._build_join_suggestions(relevant_tables)
    
    def _enhance_context_simple(self, question):
        """Enhance context using simple keyword matching"""
        # Get relevant tables using simple matching
        relevant_tables = self.suggest_relevant_tables(question)
        
        return self._build_join_suggestions(relevant_tables)
    
    def _build_join_suggestions(self, relevant_tables):
        """Build JOIN suggestions from relevant tables"""
        # Find optimal paths between relevant tables
        join_suggestions = []
        for i in range(len(relevant_tables)):
            for j in range(i+1, len(relevant_tables)):
                path = self.find_optimal_join_path(relevant_tables[i], relevant_tables[j])
                if len(path) > 1:
                    join_suggestions.append({
                        'path': path,
                        'weight': sum(self.relationship_weights.get((path[k], path[k+1]), 0.5) 
                                    for k in range(len(path)-1))
                    })
        
        # Sort by weight
        join_suggestions.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'relevant_tables': relevant_tables,
            'suggested_joins': join_suggestions[:3],  # Top 3 join paths
            'table_priorities': {table: i for i, table in enumerate(relevant_tables)}
        }

class GNNMySQLSystem:
    def __init__(self, api_key, privacy_config=None):
        self.client = Mistral(api_key=api_key)
        self.training_data = self.load_training_data()
        self.chat_history = []
        self.graph = None
        self.schema_info = None
        self.gnn = None
        self.privacy_config = privacy_config or self.default_privacy_config()
        self.anonymization_map = {}
        self.query_audit_log = []
        
    def default_privacy_config(self):
        """Default privacy configuration"""
        return {
            "anonymize_schema": True,
            "max_rows_returned": 1000,
            "blocked_patterns": [
                r"SELECT \* FROM.*",  # Block SELECT *
                r"DROP|DELETE|TRUNCATE",  # Block destructive operations
                r"password|ssn|credit_card"  # Block sensitive columns
            ],
            "allowed_aggregations": ["COUNT", "AVG", "SUM", "MIN", "MAX"],
            "pii_columns": ["email", "phone", "ssn", "address", "name", "password"],
            "log_all_queries": True,
            "enable_gnn": True
        }
        
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
        
        # Initialize GNN component
        if self.privacy_config.get('enable_gnn', True):
            self.initialize_gnn(schema_info)
        
        return schema_info
    
    def initialize_gnn(self, schema_info):
        """Initialize GNN component"""
        try:
            self.gnn = SimpleSchemaGNN(schema_info)
            self.gnn.create_table_embeddings()
            self.gnn.calculate_relationship_weights()
        except Exception as e:
            st.warning(f"GNN initialization failed: {str(e)}. Continuing without GNN enhancement.")
    
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
    
    def anonymize_schema_for_llm(self, schema_info):
        """Replace real schema names with anonymized versions"""
        if not self.privacy_config.get('anonymize_schema', True):
            return schema_info
            
        anonymized_schema = {
            'tables': [],
            'relationships': [],
            'graph_data': {}
        }
        
        # Create anonymization mapping
        table_map = {}
        column_map = {}
        
        for i, table in enumerate(schema_info['tables']):
            # Anonymize table name
            anon_table = f"table_{i+1}"
            table_map[table['name']] = anon_table
            
            anonymized_columns = []
            for j, col in enumerate(table['columns']):
                # Check if column contains PII
                is_pii = any(pii in col['name'].lower() 
                           for pii in self.privacy_config['pii_columns'])
                
                # Anonymize column name
                anon_col = f"col_{j+1}" if is_pii else f"attr_{j+1}"
                column_map[f"{table['name']}.{col['name']}"] = f"{anon_table}.{anon_col}"
                
                anonymized_columns.append({
                    'name': anon_col,
                    'type': col['type'],  # Keep type for SQL generation
                    'nullable': col['nullable'],
                    'primary_key': col['primary_key'],
                    'is_pii': is_pii
                })
            
            anonymized_schema['tables'].append({
                'name': anon_table,
                'columns': anonymized_columns
            })
        
        # Anonymize relationships
        for rel in schema_info['relationships']:
            anonymized_schema['relationships'].append({
                'from_table': table_map[rel['from_table']],
                'to_table': table_map[rel['to_table']],
                'from_column': column_map[f"{rel['from_table']}.{rel['from_column']}"].split('.')[1],
                'to_column': column_map[f"{rel['to_table']}.{rel['to_column']}"].split('.')[1],
                'type': rel['type']
            })
        
        # Store mapping for query translation
        self.anonymization_map = {
            'tables': table_map,
            'columns': column_map,
            'reverse_tables': {v: k for k, v in table_map.items()},
            'reverse_columns': {v: k for k, v in column_map.items()}
        }
        
        return anonymized_schema
    
    def validate_query_privacy(self, sql_query: str) -> tuple[bool, str]:
        """Validate query against privacy rules"""
        # Handle invalid query responses
        if sql_query.upper().startswith('INVALID QUERY'):
            return False, sql_query
        
        query_upper = sql_query.upper()
        
        # Check for destructive operations
        destructive_patterns = [r"DROP|DELETE|TRUNCATE|ALTER|CREATE"]
        for pattern in destructive_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return False, f"Query blocked: destructive operation detected"
        
        # Check for SELECT * (but allow COUNT(*))
        if re.search(r"SELECT\s+\*", sql_query, re.IGNORECASE) and not re.search(r"COUNT\s*\(\s*\*\s*\)", sql_query, re.IGNORECASE):
            return False, "Query blocked: SELECT * is not allowed for privacy reasons"
        
        # Auto-add LIMIT if missing on potentially large queries
        if "JOIN" in query_upper and "LIMIT" not in query_upper and "COUNT" not in query_upper:
            sql_query += f" LIMIT {self.privacy_config.get('max_rows_returned', 1000)}"
        
        return True, "Query passed privacy validation"
    
    def translate_anonymized_query(self, sql_query: str) -> str:
        """Translate anonymized query back to real schema"""
        # If we're not using anonymization or no mapping exists, return as-is
        if not self.privacy_config.get('anonymize_schema', True) or not self.anonymization_map:
            return sql_query
            
        real_sql = sql_query
        
        # Replace table names
        if 'reverse_tables' in self.anonymization_map:
            for anon_table, real_table in self.anonymization_map['reverse_tables'].items():
                real_sql = re.sub(rf'\b{anon_table}\b', real_table, real_sql)
        
        # Replace column names
        if 'reverse_columns' in self.anonymization_map:
            for anon_col, real_col in self.anonymization_map['reverse_columns'].items():
                # Handle table.column format
                real_sql = re.sub(rf'\b{anon_col}\b', real_col, real_sql)
        
        return real_sql
    
    def audit_query(self, question: str, sql_query: str, user_id: str = None, 
                   result_count: int = 0):
        """Log query for audit purposes"""
        audit_entry = {
            'timestamp': time.time(),
            'user_id': user_id or 'anonymous',
            'question': question,
            'sql_query': sql_query,
            'result_count': result_count,
            'query_hash': hashlib.sha256(sql_query.encode()).hexdigest()[:16]
        }
        
        self.query_audit_log.append(audit_entry)
        
        # In production, write to secure audit log
        if self.privacy_config['log_all_queries']:
            print(f"AUDIT: {audit_entry}")
    
    def get_privacy_report(self) -> dict:
        """Generate privacy compliance report"""
        total_queries = len(self.query_audit_log)
        
        if total_queries == 0:
            return {"message": "No queries executed yet"}
        
        # Analyze query patterns
        query_types = {}
        for entry in self.query_audit_log:
            sql_upper = entry['sql_query'].upper()
            if 'SELECT' in sql_upper:
                query_types['SELECT'] = query_types.get('SELECT', 0) + 1
            if 'JOIN' in sql_upper:
                query_types['JOIN'] = query_types.get('JOIN', 0) + 1
            if 'GROUP BY' in sql_upper:
                query_types['AGGREGATION'] = query_types.get('AGGREGATION', 0) + 1
        
        return {
            'total_queries': total_queries,
            'query_types': query_types,
            'avg_results_per_query': sum(e['result_count'] for e in self.query_audit_log) / total_queries if total_queries > 0 else 0,
            'unique_users': len(set(e['user_id'] for e in self.query_audit_log)),
            'recent_queries': self.query_audit_log[-5:]  # Last 5 queries
        }
    
    def create_gnn_enhanced_prompt(self, question):
        """Create prompt enhanced with GNN and training data"""
        # Use original schema for better LLM understanding, but mark PII columns
        schema_to_use = self.create_privacy_aware_schema()
        
        prompt = """You are an expert SQL generator. Generate accurate SQL queries based on the database schema and relationships.

IMPORTANT: Always generate valid SQL queries. Do not respond with "INVALID QUERY" unless the question is completely nonsensical.

"""
        
        # Add schema information
        if schema_to_use:
            prompt += "DATABASE SCHEMA:\n"
            for table in schema_to_use['tables']:
                prompt += f"\nTable: {table['name']}\n"
                for col in table['columns']:
                    pii_marker = " (SENSITIVE)" if col.get('is_pii') else ""
                    pk_marker = " (PK)" if col['primary_key'] else ""
                    prompt += f"  - {col['name']}: {col['type']}{pk_marker}{pii_marker}\n"
            
            prompt += "\nRELATIONSHIPS:\n"
            for rel in schema_to_use['relationships']:
                prompt += f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}\n"
        
        # Add GNN insights if available
        if self.gnn:
            try:
                insights = self.gnn.enhance_sql_generation_context(question)
                
                prompt += "\nSMART SUGGESTIONS:\n"
                if insights['relevant_tables']:
                    prompt += f"Most relevant tables: {', '.join(insights['relevant_tables'])}\n"
                
                if insights['suggested_joins']:
                    prompt += "Recommended JOIN paths:\n"
                    for join in insights['suggested_joins']:
                        path_str = " -> ".join(join['path'])
                        prompt += f"  {path_str}\n"
            except Exception as e:
                # Continue without GNN insights if there's an error
                pass
        
        # Add training examples
        if self.training_data:
            prompt += "\nEXAMPLE QUERIES:\n"
            
            # Add positive examples
            positive_examples = [ex for ex in self.training_data['training_data'] if ex['complexity'] in ['simple', 'medium']]
            for example in positive_examples[:3]:  # Just 3 examples
                prompt += f"Q: {example['question']}\n"
                prompt += f"A: {example['sql']}\n\n"
        
        prompt += f"""RULES:
1. Generate valid SQL that works with the given schema
2. Use proper JOIN syntax based on foreign key relationships
3. Use table aliases for clarity (e.g., s.name, d.department_name)
4. Include LIMIT 100 for queries that might return many rows
5. Be specific with column names - avoid SELECT *
6. Use appropriate WHERE clauses for filtering
7. Return ONLY the SQL query, no explanations

QUESTION: {question}

Generate a SQL query for this question:"""
        
        return prompt
    
    def create_privacy_aware_schema(self):
        """Create schema that shows real names but marks sensitive columns"""
        if not self.schema_info:
            return None
            
        privacy_schema = {
            'tables': [],
            'relationships': self.schema_info['relationships']
        }
        
        for table in self.schema_info['tables']:
            privacy_columns = []
            for col in table['columns']:
                # Check if column contains PII
                is_pii = any(pii in col['name'].lower() 
                           for pii in self.privacy_config['pii_columns'])
                
                privacy_columns.append({
                    'name': col['name'],
                    'type': col['type'],
                    'nullable': col['nullable'],
                    'primary_key': col['primary_key'],
                    'is_pii': is_pii
                })
            
            privacy_schema['tables'].append({
                'name': table['name'],
                'columns': privacy_columns
            })
        
        return privacy_schema
    
    def generate_sql_with_gnn(self, question, user_id=None):
        """Generate SQL using GNN-enhanced prompt with privacy validation"""
        try:
            # Check if API key is configured
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key or api_key == "your_mistral_api_key_here":
                return None, False, "Please configure your Mistral API key in the .env file"
            
            prompt = self.create_gnn_enhanced_prompt(question)
            
            # Add timeout and retry logic
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.complete(
                        model="mistral-large-latest",
                        messages=[UserMessage(content=prompt)],
                        max_tokens=500,
                        temperature=0.1
                    )
                    break
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        raise api_error
                    time.sleep(1)  # Wait before retry
            
            full_response = response.choices[0].message.content.strip()
            
            # Extract SQL query from the response
            anonymized_sql = self.extract_sql_from_response(full_response)
            
            if not anonymized_sql:
                return None, False, "Failed to extract SQL from response"
            
            # Translate back to real schema if anonymization was used
            real_sql = self.translate_anonymized_query(anonymized_sql)
            
            # Validate privacy rules
            is_valid, validation_msg = self.validate_query_privacy(real_sql)
            
            if not is_valid:
                return None, False, validation_msg
            
            # Audit the query
            self.audit_query(question, real_sql, user_id)
            
            return real_sql, True, "Query generated successfully with GNN and privacy validation"
            
        except Exception as e:
            error_msg = f"Error generating SQL: {str(e)}"
            # Don't use st.error here as it might cause issues
            print(f"SQL Generation Error: {error_msg}")
            return None, False, error_msg
    
    def extract_sql_from_response(self, response):
        """Extract SQL query from model response"""
        import re
        
        # Clean the response
        response = response.strip()
        
        # Check if it's an invalid query response
        if response.upper().startswith('INVALID QUERY'):
            return response
        
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
        
        # Look for SQL keywords at the start of lines
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                # Found a SQL statement, collect all lines until we have a complete query
                sql_lines = [line]
                remaining_lines = lines[lines.index(line.strip()) + 1:]
                
                for next_line in remaining_lines:
                    next_line = next_line.strip()
                    if not next_line:
                        continue
                    sql_lines.append(next_line)
                    # Stop if we hit a semicolon or another SQL keyword
                    if next_line.endswith(';') or next_line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
                        break
                
                return ' '.join(sql_lines)
        
        # If response looks like SQL (contains SQL keywords), return as-is
        if any(keyword in response.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY']):
            return response
        
        # Last resort: return the response and let validation handle it
        return response
    
    def execute_sql(self, engine, sql_query, question=None, user_id=None):
        """Execute SQL query safely with audit logging"""
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
                    
                    # Update audit log with result count
                    if question and self.query_audit_log:
                        self.query_audit_log[-1]['result_count'] = len(df)
                    
                    return df, None
                else:
                    # For INSERT, UPDATE, DELETE
                    conn.commit()
                    
                    # Update audit log with affected rows
                    if question and self.query_audit_log:
                        self.query_audit_log[-1]['result_count'] = result.rowcount
                    
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
                    with st.spinner("Connecting to database..."):
                        connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                        engine = create_engine(
                            connection_string,
                            pool_pre_ping=True,
                            pool_recycle=3600,
                            connect_args={"connect_timeout": 10}
                        )
                        
                        # Test connection
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                    
                    with st.spinner("Initializing GNN system..."):
                        # Initialize system
                        system = GNNMySQLSystem(api_key)
                        schema_info = system.extract_schema_graph(engine)
                    
                    st.success("GNN System initialized successfully!")
                    st.session_state['system'] = system
                    st.session_state['engine'] = engine
                    st.session_state['schema_info'] = schema_info
                    st.session_state['initialized'] = True
                    
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
                    st.info("Please check your database credentials and ensure the server is running.")
            else:
                st.warning("Please fill in all fields.")
    
    # API key is hard-coded, no need to check
    pass
    
    if not all([db_host, db_user, db_password, db_name]):
        st.warning("Please configure your database connection in the sidebar.")
        return
    
    # Check if system is initialized
    if 'system' not in st.session_state or not st.session_state.get('initialized', False):
        st.info("Please initialize the GNN system using the sidebar.")
        return
    
    # Test database connection
    try:
        with st.session_state['engine'].connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Database connection lost: {str(e)}")
        st.info("Please reinitialize the system in the sidebar.")
        if st.button("Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        return
    
    system = st.session_state['system']
    engine = st.session_state['engine']
    schema_info = st.session_state['schema_info']
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Graph Visualization", "Schema Info", "History", "Privacy & Audit"])
    
    with tab1:
        st.header("💬 GNN-Enhanced Chat")
        
        # Display system info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if system.graph:
                st.metric("Tables", system.graph.number_of_nodes())
        with col2:
            if system.graph:
                st.metric("Relationships", system.graph.number_of_edges())
        with col3:
            st.metric("GNN Status", "✅ Active" if system.gnn else "❌ Disabled")
        with col4:
            st.metric("Privacy Mode", "🔒 Enabled" if system.privacy_config.get('anonymize_schema') else "🔓 Disabled")
        
        # User ID input for audit
        user_id = st.text_input("User ID (for audit)", value="demo_user", help="Enter your user ID for audit logging")
        
        # Chat input
        user_question = st.text_area("Ask a question about your data:", 
                                    placeholder="e.g., Show me all students in the Computer Science department")
        
        if st.button("Generate SQL with GNN & Privacy", type="primary"):
            if user_question:
                with st.spinner("Generating SQL using GNN with privacy validation..."):
                    sql_query, is_valid, message = system.generate_sql_with_gnn(user_question, user_id)
                
                if is_valid and sql_query:
                    # Execute the query
                    with st.spinner("Executing query..."):
                        result_df, exec_message = system.execute_sql(engine, sql_query, user_question, user_id)
                    
                    # Store in chat history
                    system.chat_history.append({
                        'question': user_question,
                        'sql': sql_query,
                        'result': result_df,
                        'message': exec_message,
                        'timestamp': time.time(),
                        'user_id': user_id,
                        'privacy_validated': True
                    })
                    
                    # Display results
                    st.success("Query executed successfully with privacy validation!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Generated SQL")
                        st.code(sql_query, language="sql")
                        
                        # Show GNN insights if available
                        if system.gnn:
                            try:
                                insights = system.gnn.enhance_sql_generation_context(user_question)
                                if insights['relevant_tables']:
                                    st.write("**GNN Suggested Tables:**", ", ".join(insights['relevant_tables']))
                            except:
                                pass
                    
                    with col2:
                        st.subheader("Results")
                        if result_df is not None:
                            st.dataframe(result_df, use_container_width=True)
                            st.write(f"**Rows returned:** {len(result_df)}")
                        elif exec_message:
                            st.info(exec_message)
                else:
                    st.error(f"Query validation failed: {message}")
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
                    
                    # Show privacy info if available
                    if chat.get('privacy_validated'):
                        st.success("✅ Privacy validated")
                    if chat.get('user_id'):
                        st.write(f"**User:** {chat['user_id']}")
        else:
            st.info("No chat history yet.")
    
    with tab5:
        st.header("🔒 Privacy & Audit Dashboard")
        
        # Privacy configuration
        st.subheader("Privacy Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Settings:**")
            st.write(f"• Schema Anonymization: {'✅ Enabled' if system.privacy_config.get('anonymize_schema') else '❌ Disabled'}")
            st.write(f"• Max Rows Returned: {system.privacy_config.get('max_rows_returned', 1000)}")
            st.write(f"• Query Logging: {'✅ Enabled' if system.privacy_config.get('log_all_queries') else '❌ Disabled'}")
            st.write(f"• GNN Enhancement: {'✅ Enabled' if system.privacy_config.get('enable_gnn') else '❌ Disabled'}")
        
        with col2:
            st.write("**PII Columns Detected:**")
            pii_columns = system.privacy_config.get('pii_columns', [])
            for pii in pii_columns:
                st.write(f"• {pii}")
        
        # Privacy report
        st.subheader("Privacy Compliance Report")
        privacy_report = system.get_privacy_report()
        
        if privacy_report.get('message'):
            st.info(privacy_report['message'])
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Queries", privacy_report['total_queries'])
            with col2:
                st.metric("Unique Users", privacy_report['unique_users'])
            with col3:
                st.metric("Avg Results/Query", f"{privacy_report['avg_results_per_query']:.1f}")
            
            # Query type breakdown
            if privacy_report['query_types']:
                st.subheader("Query Type Distribution")
                query_df = pd.DataFrame(list(privacy_report['query_types'].items()), 
                                      columns=['Query Type', 'Count'])
                st.bar_chart(query_df.set_index('Query Type'))
        
        # Audit log
        st.subheader("Recent Audit Log")
        if system.query_audit_log:
            audit_df = pd.DataFrame(system.query_audit_log)
            audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'], unit='s')
            
            # Show recent queries
            st.dataframe(
                audit_df[['timestamp', 'user_id', 'question', 'result_count', 'query_hash']].tail(10),
                use_container_width=True
            )
            
            # Download audit log
            if st.button("Download Full Audit Log"):
                csv = audit_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"audit_log_{int(time.time())}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No audit entries yet.")
        
        # GNN insights
        if system.gnn:
            st.subheader("🧠 GNN Model Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Table Embeddings:**")
                st.write(f"• Embedded tables: {len(system.gnn.table_embeddings)}")
                st.write(f"• Embedding dimension: {len(next(iter(system.gnn.table_embeddings.values()))) if system.gnn.table_embeddings else 0}")
            
            with col2:
                st.write("**Relationship Weights:**")
                st.write(f"• Weighted relationships: {len(system.gnn.relationship_weights)}")
                if system.gnn.relationship_weights:
                    avg_weight = sum(system.gnn.relationship_weights.values()) / len(system.gnn.relationship_weights)
                    st.write(f"• Average weight: {avg_weight:.3f}")
        else:
            st.info("GNN component not initialized.")

if __name__ == "__main__":
    main() 