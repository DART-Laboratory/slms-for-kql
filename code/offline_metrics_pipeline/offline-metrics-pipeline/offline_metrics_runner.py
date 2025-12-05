# Imports:
from pythonnet import load

try:
    load('mono')
except:
    load('coreclr')

import clr
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from System import Reflection

base_path = Path(__file__).resolve().parent
file_path = f"{base_path}/Kusto.Language.dll"
Reflection.Assembly.LoadFile(file_path)

from Kusto.Language import KustoCode, GlobalState
from Kusto.Language.Symbols import DatabaseSymbol, TableSymbol, ColumnSymbol, ScalarTypes
from typing import Set, Dict, List, Literal
import regex as re
import yaml

from query_dataset import QueryDataset
from helpers import *

print("Importing Packages Finished...")

# Results should go in the following folder as defined here:

output_folder = sys.argv[2]
output_file = sys.argv[3]

# Loading in Additional Information:
with open(f'{base_path}/schema/defender.yml', 'r') as file:
    defender_information = yaml.safe_load(file)

with open(f'{base_path}/schema/sentinel.yml', 'r') as file:
    sentinel_information = yaml.safe_load(file)
    
valid_tables = set(defender_information.keys())

# Run the Offline:
results = []
files_of_interest = sys.argv[1]

repo = QueryDataset(f'{files_of_interest}')
all_queries = repo.get_all_queries()

for query in all_queries:
    query_id, prompt, connector, baseline, llmResult = repo.get_query_details(query)
    namespace = create_kusto_namespace()
    setup_global_state("Defender", namespace)
    global_state = namespace.get('global_state')
    metrics = {
        'id': query_id,
        'prompt': prompt,
        'connector': "Defender",
        'syntax_score': syntax_score(llmResult),
        'semantic_score': semantic_score(llmResult, global_state),
        'table_score': table_score(baseline, llmResult, valid_tables),
        'filter_column_score': filter_col_score(baseline, llmResult, valid_tables, defender_information, connector),
        'filter_literal_score': filter_literal_score(baseline, llmResult)
    }
    results.append(metrics)

df = pd.DataFrame(results)

os.makedirs(f'{output_folder}',exist_ok=True)
full_path = os.path.join(f'{output_folder}', output_file)
full_path_averages = full_path.replace('.csv', '_averages.csv')
print(f'Output Saved to {full_path}')
print(f'Output Saved to {full_path_averages}')

# average of the data:
avg_df = df.drop(['id', 'prompt', 'connector'], axis=1).mean(numeric_only=True).to_frame().T
avg_df.to_csv(full_path_averages, index=False)
df.to_csv(full_path, index=False)

# Latex Format:
print("-------------------------------------")
print("LATEX FORMAT FOR EASY COPY AND PASTE:")
print(f"{round(avg_df.loc[0]['syntax_score'], 3)} & {round(avg_df.loc[0]['semantic_score'], 3)} & {round(avg_df.loc[0]['table_score'], 3)} & {round(avg_df.loc[0]['filter_column_score'], 3)} & {round(avg_df.loc[0]['filter_literal_score'], 3)}")