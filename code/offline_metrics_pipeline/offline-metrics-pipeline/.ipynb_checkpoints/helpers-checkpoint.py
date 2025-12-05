from pythonnet import load

try:
    load('coreclr')
except:
    load('mono')
    
import clr
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from System import Reflection
from pathlib import Path

base_path = Path(__file__).resolve().parent
file_path = f"{base_path}/Kusto.Language.dll"
Reflection.Assembly.LoadFile(file_path)

from Kusto.Language import KustoCode, GlobalState
from Kusto.Language.Symbols import DatabaseSymbol, TableSymbol, ColumnSymbol, ScalarTypes
from typing import Set, Dict, List, Literal
import regex as re
import yaml

from difflib import get_close_matches

clr.AddReference("Kusto.Language")

def setup_diagnostic_storage():
    """Initialize storage lists for diagnostic results"""
    return {
        'diagnostic_counts': [],
        'severity': [],
        'messages': [],
        'troublemakers': []
    }

def create_kusto_namespace():
    """Create namespace with required Kusto components"""
    return {
        'ColumnSymbol': ColumnSymbol,
        'ScalarTypes': ScalarTypes,
        'TableSymbol': TableSymbol,
        'DatabaseSymbol': DatabaseSymbol,
        'GlobalState': GlobalState
    }

def process_diagnostics(diagnostics, kql_query):
    """Process diagnostic results for a single query"""
    query_severity = []
    query_messages = []
    query_troublemakers = []
    
    for diag in diagnostics:
        query_severity.append(str(diag.Severity))
        query_messages.append(diag.Message)
        query_troublemakers.append(kql_query[diag.Start:diag.End])
        
    return diagnostics.Count, query_severity, query_messages, query_troublemakers

def handle_query_error():
    """Return error state for failed query analysis"""
    return -1, ['Error'], ['Query analysis failed'], ['']

def syntax_score(query: str) -> int:
    """
    Syntax Score: Returns 1 if the query is syntactically correct, 0 otherwise.
    """

    if query.strip() == "":
        return 0
    try:
        code = KustoCode.Parse(query)
        return 1 if code.GetDiagnostics().Count == 0 else 0
    except Exception:
        return 0

def semantic_score(query: str, global_state) -> int:
    """
    Semantic Score: Returns 1 if no semantic errors are reported by the parser, 0 otherwise.
    """
    if query.strip() == "":
        return 0
    try:
        code = KustoCode.ParseAndAnalyze(query, global_state)
        return 1 if code.GetDiagnostics().Count == 0 else 0
    except Exception:
        return 0

def setup_global_state(connector: Literal["Defender", "Sentinel"], namespace):
    """Load appropriate schema based on connector type"""
    column_defs = f"{base_path}/schema/{connector.lower()}_columns.txt"
    
    with open(column_defs, 'r') as f:
        content = f.read()
        exec(content, namespace)
    
    global_state = namespace['global_state']

def extract_tables(query: str, valid_tables: List) -> Set[str]:
    """Extract tables using combined Defender/Sentinel schema"""
    # Case-insensitive patterns for KQL table references
    patterns = [
        r'\b(from|join|union)\s+([a-zA-Z_]+)',  # Standard references
        r'let\s+([a-zA-Z_]+)\s*=',              # CTE definitions
        r'invoke\s+([a-zA-Z_]+)\s*\(',          # Function calls
        r'\[?\b([a-zA-Z_]+)\]?\s*\|',           # Piped tables
        r'\(([a-zA-Z_]+)\s+\|'                  # Subqueries
    ]
    
    tables = set()
    normalized_query = query.lower()
    
    # Pattern matching with schema validation
    for pattern in patterns:
        for match in re.findall(pattern, normalized_query, re.IGNORECASE):
            candidate = match[-1].strip('[]')  # Handle quoted identifiers
            # Fuzzy match against schema
            matches = get_close_matches(candidate, valid_tables, n=1, cutoff=0.9)
            if matches:
                tables.add(matches[0])
    
    # First token heuristic (NL2KQL baseline)
    first_token = re.search(r'^\s*([a-zA-Z_]+)', query)
    if first_token:
        ft = first_token.group(1)
        if ft in valid_tables:
            tables.add(ft)
    
    return tables

def extract_referenced_tables(kql_query, tables):
    """Determines which tables are referenced in a KQL query"""
    
    referenced_tables = []
    for entry in tables:
        if entry in kql_query:
            referenced_tables.append(entry)
            
    return referenced_tables

def table_score(baseline: str, llmResult: str, valid_tables: List) -> float:
    """
    Computes table score per NL2KQL paper specifications.
    
    Score = |T_baseline ∩ T_generated| / |T_generated| 
            if T_baseline ⊆ T_generated, else 0
    
    Uses string-based table extraction from queries.
    """
    
    # Extract tables from both queries
    baseline_tables = extract_tables(baseline, valid_tables)
    llmResult_tables = extract_tables(llmResult, valid_tables)
    
    # Convert to sets for subset check
    base_set = set(baseline_tables)
    result_set = set(llmResult_tables)
    
    # Implement paper's subset condition
    if not base_set.issubset(result_set):
        return 0.0
    
    # Calculate intersection ratio
    intersection = base_set & result_set
    if intersection:
        return len(intersection) / len(result_set)
    return 0

def extract_columns(kql_query, columns):
    """Determines which columns are referenced in a KQL query"""
    
    referenced_columns = []
    for entry in columns:
        if entry in kql_query:
            referenced_columns.append(entry)
            
    return referenced_columns

def jaccard_similarity(a, b):
    """Computes the Jaccard Similarity between two sets"""
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    if union and intersection:
        return len(intersection)/len(union)
    return 0
               
def filter_col_score(baseline, llmResult, valid_tables, information, connector: Literal["Defender", "Sentinel"]):
    """Compute the filter column score between two KQL queries (as defined by NL2KQL)"""
    
    baseline_tables = extract_tables(baseline, valid_tables)
    llmResult_tables = extract_tables(llmResult, valid_tables)
    
    baseline_cols = []
    llmResult_cols = []
    if connector not in ["Defender", "Sentinel"]:
        TypeError("Error: Connector must be 'Defender' or 'Sentinel'")
    if connector == "Defender":
        for table in baseline_tables:
            baseline_cols += information[table]['Columns']
        baseline_cols = [column for column in baseline_cols if column in baseline]

        for table in llmResult_tables:
            if table in information:
                llmResult_cols += information[table]['Columns']
            else:
                # if the table is not part of defender, then we have to check the entire schema
                # not ideal - but resolves of case of table wrong columns right
                for key in information.keys():
                    llmResult_cols += information[key]['Columns']
        llmResult_cols = [column for column in llmResult_cols if column in llmResult]
        
    if connector == "Sentinel":
        for table in baseline_tables:
                baseline_cols += information[table]['Columns']
        baseline_cols = [column for column in baseline_cols if column in baseline]

        for table in llmResult_tables:
            if table in information:
                llmResult_cols += information[table]['Columns']
            else:
                # if the table is not part of sentinel, then we have to check the entire schema
                # not ideal - but resolves of case of table wrong columns right
                for key in information.keys():
                    llmResult_cols += information[key]['Columns']
        llmResult_cols = [column for column in llmResult_cols if column in llmResult]

    return jaccard_similarity(set(baseline_cols), set(llmResult_cols))

def filter_literal_score(baseline, llm_query):

    code_baseline = KustoCode.Parse(baseline)
    code_llm = KustoCode.Parse(llm_query)

    tokens_llm = []
    tokens_gt = []

    for entry in code_baseline.GetLexicalTokens():
        if 'literal' in str(entry.Kind) or 'Literal' in str(entry.Kind):
            tokens_gt.append(entry.Text) 
    
    for entry in code_llm.GetLexicalTokens():
        if 'literal' in str(entry.Kind) or 'Literal' in str(entry.Kind):
            tokens_llm.append(entry.Text)

    tokens_llm_new = [re.sub(r'"|\'', '', entry) for entry in tokens_llm]
    tokens_gt_new = [re.sub(r'"|\'', '', entry) for entry in tokens_gt]
    
    return jaccard_similarity(set(tokens_llm_new), set(tokens_gt_new))