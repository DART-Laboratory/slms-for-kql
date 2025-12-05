# Import the Necessary Packages:
from pythonnet import load

try:
    load('mono')
except:
    load('coreclr')

import clr
import re
import os
import yaml
import json
import pickle
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from System import Reflection

from google import genai as genai_client
from google.genai.types import GenerateContentConfig
from sklearn.metrics.pairwise import cosine_similarity

relevant_path = Path(__file__).resolve().parent
file_path = f"{relevant_path}/Kusto.Language.dll"
Reflection.Assembly.LoadFile(file_path)

from Kusto.Language import KustoCode
from Kusto.Language.Symbols import DatabaseSymbol, TableSymbol, ColumnSymbol, ScalarTypes
from Kusto.Language import GlobalState
from System.Collections.Generic import HashSet
from Kusto.Language.Syntax import SyntaxElement
from System import Action

from typing import Set, Dict, List, Literal

with open("../config.yaml", "r") as f:
    token_config = yaml.safe_load(f)

API_KEY = token_config['genai']['token']
client = genai_client.Client(api_key=API_KEY)

print("Importing all packages...")

# Read in the results generated with the help of NL2KQL:
def extract_tables(kql_query: str, tables: List) -> List:
    """Determines which tables are referenced in a KQL query"""
    
    referenced_tables = []
    for entry in tables:
        if entry in kql_query:
            referenced_tables.append(entry)
            
    return referenced_tables

def create_kusto_namespace():
    """Create namespace with required Kusto components"""
    return {
        'ColumnSymbol': ColumnSymbol,
        'ScalarTypes': ScalarTypes,
        'TableSymbol': TableSymbol,
        'DatabaseSymbol': DatabaseSymbol,
        'GlobalState': GlobalState
    }

def setup_global_state(connector: Literal["Defender", "Sentinel"], namespace):
    """Load appropriate schema based on connector type"""
    column_defs = f"{relevant_path}/{connector.lower()}_columns.txt"
    
    with open(column_defs, 'r') as f:
        content = f.read()
        exec(content, namespace)
        
    global_state = namespace['global_state']

def get_database_table_columns(code):
    columns = HashSet[ColumnSymbol]()
    
    def add_database_table_columns(column, globals, columns_set):
        if globals.GetTable(column) is not None:
            columns_set.Add(column)
    
    def process_node(n):
        if isinstance(n.ReferencedSymbol, ColumnSymbol):
            add_database_table_columns(n.ReferencedSymbol, code.Globals, columns)
    
    # Convert the Python function to a C# Action<SyntaxNode>
    action = Action[SyntaxElement](process_node)
    SyntaxElement.WalkNodes(code.Syntax, action)
    return columns
    
with open(f"{relevant_path}/Defender_Schema.json") as f:
    defender_information = yaml.safe_load(f)

table_lst = [entry['Table'] for entry in defender_information[0]['Tables']]

file_path = sys.argv[1]
model_name = sys.argv[2]
destination = sys.argv[3]

df = pd.read_csv(file_path)
if 'id' in df.columns:
    df = df.drop('id', axis=1)

llm_kql = list(df['LLM-KQL-Extracted'])

errors = []
counter = 1

for entry in llm_kql:

    query = f"""{entry}"""

    # Change this later, because this could contain multiple tables:
    tables = extract_tables(query, table_lst)
    cols = []
    tables_revised = []

    # Get the respective columns for each table:
    # for table_mention in tables:
    #     for entry in defender_information[0]['Tables']:
    #         if entry['Table'] == table_mention:
    #             cols = entry['Columns']

    #     columns = []
    #     type_match = {'System.DateTime': ScalarTypes.DateTime, 'System.String': ScalarTypes.String, 'Boolean': ScalarTypes.Bool, 'Double': ScalarTypes.Decimal, 'System.Int32': ScalarTypes.Int, 'System.Int64': ScalarTypes.Int, 'System.Object': ScalarTypes.Dynamic}
    
    #     for entry in cols:
    #         columns.append(ColumnSymbol(entry['Name'], type_match[entry['Type']]))
        
        # for entry in tables:
        #     tables_revised.append(TableSymbol(entry, columns))

    # global_state = GlobalState.Default.WithDatabase(DatabaseSymbol('Defender', tables_revised))

    namespace = create_kusto_namespace()
    setup_global_state("Defender", namespace)
    global_state = namespace.get('global_state')
    code = KustoCode.ParseAndAnalyze(query, global_state)
    
    #ref_cols = get_database_table_columns(code)

    diagnostics = code.GetDiagnostics()
    if (diagnostics.Count > 0):
        for diag in diagnostics:
            raw_msg = str(diag.Message)
            safe_msg = raw_msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        
            start = int(diag.Start)
            end = int(diag.End)
            if start < 0:
                start = 0
            if end > len(query):
                end = len(query)
        
            message = {
                'id': counter,
                'query': query, 
                'severity': str(diag.Severity),
                'message': safe_msg,
                'length': int(diag.Length),
                'start': start,
                'end': end,
                'troublemaker': query[start:end] if 0 <= start < end <= len(query) else ""
            }
            errors.append(message)
            
    counter = counter + 1

with open(f'{relevant_path}/Defender_DataCatalog.yml') as f:
    data = yaml.safe_load(f)

with open(f'{relevant_path}/resulting_dict.pkl', 'rb') as f:
    resulting_dict = pickle.load(f)

for item in errors:
    item['id'] -= 1

errors_df = pd.DataFrame(errors)[['id', 'message']]
aggregated_errors_df = errors_df.groupby('id')['message'].agg(list).reset_index()
df = df.rename(columns = {'Unnamed: 0': 'id'})
merged_df = pd.merge(df, aggregated_errors_df, on='id', how='left')

# Replace NaNs with empty lists:
merged_df['message'] = merged_df['message'].apply(lambda d: d if isinstance(d, list) else [])
merged_df['LLM-KQL-Extracted'] = merged_df['LLM-KQL-Extracted'].fillna("")

counter = 1
cond_two = 0

# Miscellaneous:
with open(f'{relevant_path}/defender.yml', 'r') as file:
    defender_information = yaml.safe_load(file)
    
table_lst = list(defender_information.keys())

# Iterate through each of the rows:
for idx, row in merged_df.iterrows():

    cond_two = 0
    # Base Case (No Errors Detected through the Semantic Check):
    if row['message'] == []:
        continue

    # Error Case (More than one error detected through the Semantic Check):
    else:
        main_embedding = -1
        final_query = row['LLM-KQL-Extracted']
        revised_query = final_query

        for err in row['message']:
            # Condition One (Finding Replacements: Undefined Identifiers):
            if re.search(r'The name \'(.*)\' does not refer to any known column, table, variable or function.', err):

                tables = [table for table in table_lst if table in final_query]
                col_of_interest = re.search(r'The name \'(.*)\' does not refer to any known column, table, variable or function.', err).group(1)
                
                # Find the table the column belongs to:
                cosine_similarities = []
                responses = []
                
                filtered_dict = {key: value for key, value in resulting_dict.items() if key in tables}
                flattened_dict = {}
                for key in filtered_dict.keys():
                    sub_filtered_dict = filtered_dict[key]
                    for subkey in sub_filtered_dict.keys():
                        flattened_dict[f"{key}, {subkey}"] = filtered_dict[key][subkey].embeddings[0].values
                
                # Pick the right table + column embedding to compare to:
                if len(tables) == 1:
                    try:
                        main_embedding = flattened_dict[f"{tables[0]}, {col_of_interest}"]
                    except:
                        for entry in data:
                            if entry['Name'] == tables[0]:
                                try:
                                    main_embedding = client.models.embed_content(model = 'text-embedding-004', contents = f"Table Description: {entry['Description']}, Column: {col_of_interest}").embeddings[0].values
                                except:
                                    time.sleep(60)
                                    main_embedding = client.models.embed_content(model = 'text-embedding-004', contents = f"Table Description: {entry['Description']}, Column: {col_of_interest}").embeddings[0].values
                                    
                elif len(tables) > 1:
                    # 1. Tokenize the LLM-KQL into separate pieces (split by space)
                    tokenized_lst = revised_query.split(" ")
                    index_loc = -1

                    for entry in tokenized_lst:
                        if col_of_interest in entry:
                            index_loc = tokenized_lst.index(entry)
                            break

                    # 2. Work backwards to find the nearest table
                    if index_loc != -1:
                        multi_table = ""
                        for b in range(index_loc, -1, -1):
                            if multi_table == "":
                                for entry in table_lst:
                                    if entry in tokenized_lst[b]:
                                        multi_table = entry
                            else:
                                break
                            
                        try:
                            main_embedding = flattened_dict[f"{multi_table}, {col_of_interest}"]
                        except:
                            for entry in data:
                                if entry['Name'] == multi_table:
                                    try:
                                        main_embedding = client.models.embed_content(model = 'text-embedding-004', contents = f"Table Description: {entry['Description']}, Column: {col_of_interest}").embeddings[0].values
                                    except:
                                        time.sleep(60)
                                        main_embedding = client.models.embed_content(model = 'text-embedding-004', contents = f"Table Description: {entry['Description']}, Column: {col_of_interest}").embeddings[0].values
                
                # Sometimes a replacement just isn't possible:
                if main_embedding == -1:
                    continue
                else:
                    cosine_similarities = []
                    for key in flattened_dict.keys():
                        cosine_similarities.append(cosine_similarity(np.array(main_embedding).reshape(1,-1), np.array(flattened_dict[key]).reshape(1,-1)))
                    
                    # Pick the top response, and make sure the cosine similarity is above 0.9:
                    if len(cosine_similarities) > 0:
                        top_idx = np.argmax(cosine_similarities)
                        if cosine_similarities[top_idx] > 0.9:
                            choice = list(flattened_dict.keys())[top_idx]
                        
                            # Replace new text with old text (needs review):
                            revised_query = final_query.replace(col_of_interest, re.search(r'.*,(.*)', choice).group(1).lstrip().rstrip())

            # Condition Two:
            if cond_two == 0:
                if re.search(r'The aggregate function .* is not defined.', err):
                    revised_query += f"\n| summarize"
                    cond_two = 1
                elif re.search(r'The aggregate function .* is not allowed in this context.', err):
                    revised_query += f"\n| summarize"
                    cond_two = 1
                elif re.search(r'The name .* is not an aggregate or scalar function.', err):
                    revised_query += f"\n| summarize"
                    cond_two = 1

            # Condition Three:
            if re.search(r'between (.*)\|\n?', revised_query):
    
                between_section = re.search(r'between (.*)\|\n?', revised_query).group(1)
                between_section_new = between_section
                    
                counter = 0
                for char in between_section_new:
                    # Increment counter for opening parenthesis
                    if char == '(':  
                        counter += 1
                        
                    # Decrement counter for closing parenthesis
                    elif char == ')':  
                        counter -= 1
                        
                # If counter goes negative, parentheses are unbalanced
                if counter < 0: 
                    add = ""
                    for i in range(-counter):
                        add += "("
                    revised_query = revised_query.replace(between_section_new, f"{add}{between_section_new}")
                elif counter > 0:
                    add = ""
                    for i in range(counter):
                        add += ")"
                    revised_query = revised_query.replace(between_section_new, f"{between_section_new}{add}")
                
                elif counter == 0:
                    if between_section.startswith('(') == False:
                        between_section_new = "(" + between_section + ")"
                        revised_query = revised_query.replace(between_section, f"{between_section_new}")
                        
            # Condition Four:
            # Add missing extend operator, but this is largely dependent on the circumstance and we choose to leave it out (as this is subjective on what column the user wants to add).
            
        
        # Check if it improves the metrics at all, if not leave it alone:
        code = KustoCode.ParseAndAnalyze(revised_query, global_state)
        diagnostics = code.GetDiagnostics()
            
        if (diagnostics.Count == 0):
            final_query = revised_query
            merged_df.at[idx, 'LLM-KQL-Extracted'] = final_query
        
        print(f"Processed Error for {idx}!")

merged_df[['NLQ', 'baseline', 'LLM-KQL-Extracted']].to_csv(f'{destination}/{model_name}-refined.csv')
results = {"queries": []}

for idx, row in merged_df.iterrows():
    results["queries"].append({
        "id": idx,
        "prompt": row["NLQ"],
        "connector": "Defender",
        "baseline": row["baseline"].strip(),
        "llmResult": row['LLM-KQL-Extracted']
})

with open(f'{destination}/{model_name}-refined.yaml', 'w') as f:
    yaml.dump(
        results, 
        f,
        sort_keys=False,
        default_style='|',
        allow_unicode=True,
        width=1000)