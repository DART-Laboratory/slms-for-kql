import re
import json
import yaml
from google import genai as genai_two
import google.generativeai as genai
from google.genai import types
from google.genai.types import GenerateContentConfig

import time
import random

from typing import List

with open("../config.yaml", "r") as f:
    token_config = yaml.safe_load(f)

API_KEY = token_config['genai']['token']
client = genai_two.Client(api_key=API_KEY)

def get_query_embedding(query: str) -> List:
    try:
        response = client.models.embed_content(model = 'text-embedding-004',
                                               contents = f"{query}")
    except:
        time.sleep(60)
        response = client.models.embed_content(model = 'text-embedding-004',
                                               contents = f"{query}")
    return response.embeddings[0].values

def generate_table_embeddings(information_dict: dict) -> dict:
    """Generates table embedding stores as proposed in NL2KQL
    
    Parameters:
    - information_dict (list[str]) - List of tables provided by either Microsoft Defender or Microsoft Sentinel
    
    Returns:
    - table_embeddings (dict) - Dictionary of Table Embeddings"""
    
    table_embeddings = dict()
    embedding_count = 0

    for table in list(information_dict.keys()):
        
        # Generating the extended summary:
        extended_summary = client.models.generate_content(model = 'models/gemini-2.0-flash',
                                                  contents = f"Table: {table} \n \
                                                  Description: {information_dict[table]['Description']} \n \
                                                  Columns: {str(information_dict[table]['Columns'])}",
                                                  config=GenerateContentConfig(
                                                      system_instruction=[
                                                          "Use the KQL table name,\
                                                          KQL table description, and KQL table schema, including \
                                                          columns, to write an extended summary of the \
                                                          table covering scenarios where this KQL table\
                                                          would be useful.",
                                                      ]
                                                  )
                                                 )
        # Get the embeddings for this summary:
        embedding = client.models.embed_content(model="text-embedding-004", contents=extended_summary)
        embedding_count += 1
        print(f"Embeddings Done: {embedding_count}")
        table_embeddings[table] = embedding
        time.sleep(5)
    
    # Return ONLY the embedding lists:
    for key in table_embeddings:
        table_embeddings[key] = table_embeddings[key].embeddings[0].values
    
    return table_embeddings

def generate_value_embeddings(information_dict: dict) -> dict:
    """Generates table embedding stores as proposed in NL2KQL
    
    Parameters:
    - information_dict (list[str]) - List of tables provided by either Microsoft Defender or Microsoft Sentinel
    
    Returns:
    - table_embeddings (dict) - Dictionary of Table Embeddings"""
    
    value_embeddings = dict()
    embedding_count = 0

    for table in list(information_dict.keys()):
        for col in list(information_dict[table]['Columns']):
            # Generating the extended summary:
            extended_summary = client.models.generate_content(model = 'models/gemini-2.0-flash',
                                                      contents = f"Table: {table} \n \
                                                      Column: {str(information_dict[table]['Columns'])}",
                                                      config=GenerateContentConfig(
                                                          system_instruction=[
                                                              "Concatenate the KQL table name and column name, and write an extended summary of its purpose."
                                                          ]
                                                      )
                                                     )
            # Get the embeddings for this summary:
            embedding = client.models.embed_content(model="text-embedding-004", contents=extended_summary)
            embedding_count += 1
            print(f"Embeddings Done: {embedding_count}")
            
            if table not in value_embeddings.keys():
                value_embeddings[table] = dict()
            value_embeddings[table][col] = embedding
            
            time.sleep(5)
    
    return value_embeddings

def generate_fsdb(themes: List[str], schema_file: str):
    
    with open("data/miscellaneous/defender_schema.json") as f:
        schema_json = json.load(f)

    num_queries = 1
    fsdb = []

    while len(fsdb) < num_queries:
        # step 1: random table sampling
        tables = []
        if random.random() < 0.7:
            # 1 table
            table_id = random.choice(list(schema_json.keys()))
            tables.append(schema_json[table_id])
        else:
            # 2 tables
            table_ids = random.sample(list(schema_json.keys()), 2)
            tables = [schema_json[id] for id in table_ids]

            # 50% chance of similar column join
            if random.random() < 0.5:
                # common column types
                col_types = {}
                for table in tables:
                    for col in table['columns']:
                        col_types[col] = col_types.get(col, 0) + 1

                common_cols = [k for k,v in col_types.items() if v > 1]
                if not common_cols:
                    continue

        # step 2: select theme
        theme = random.choice(themes)
        try:
            # step 3: generate initial KQL and NLQ
            kql1 = generate_kql(tables, theme)
            time.sleep(4.5)

            # step 4: generate nlq
            nlq = generate_nlq(kql1)
            time.sleep(4.5)

            # step 5: generate secondary kql
            kql2 = generate_secondary_kql(nlq, tables)
            time.sleep(4.5)

            # step 6: check similarity
            if jaccard_similarity(kql1, kql2) < 0.5:
                continue
            
            time.sleep(5)
            exp = client.models.generate_content(model = "gemini-2.0-flash", contents = f"Explain step-by-step how the following KQL query answers the NLQ:\n\nNLQ: {nlq}\nKQL:{kql1}").text.strip()

            fsdb.append({
                "tables": [t['name'] for t in tables],
                "theme": theme,
                "nlq": nlq,
                "kql": kql1,
                "similarity_score": jaccard_similarity(kql1, kql2),
                "Explanation": exp
            })
            print(len(fsdb))

        except Exception as e:
            print(f"Error generating example: {e}")
            continue

    with open(schema_file, "w") as f:
        json.dump(fsdb, f, indent=4)

def jaccard_similarity(str1: str, str2: str) -> float:
    
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def generate_kql(tables: List, theme: str) -> str:
    """Generate initial KQL using Gemini with column context"""
    # Build table schema descriptions
    table_schemas = []
    for table in tables:
        columns = ", ".join(table['columns'])
        table_schemas.append(f"{table['name']}: [{columns}]")
    
    prompt = f"""Generate a Kusto Query Language (KQL) query using these tables and columns:\nTables and Columns: {chr(10).join(table_schemas)}\nTheme: {theme}\n\nFollow these rules:\n1. Use proper joins (if multiple tables) using existing columns\n2. Include security-relevant filters using available columns\n3. Use appropriate aggregation functions from the schema\n4. Reference only existing columns from the tables\n5. Follow KQL best practices for performance\n6. Generate a succinct KQL query following the theme\n\nReturn only the KQL query without comments"""
    
    response = client.models.generate_content(model = "gemini-2.0-flash", contents = prompt)
    return response.text.strip()

def generate_nlq(kql: str) -> str:
    """Generate natural language explanation"""
    
    prompt = f"""Explain this KQL query in imperative natural language with a maximum length of 20 words: {kql}\n\nOutput format:\n- Use concise, action-oriented language\n- Mention key tables and security concepts\n- Avoid technical jargon\n\nOutput ONLY the natural language description:"""
    
    response = client.models.generate_content(model = "gemini-2.0-flash", contents = prompt)
    return response.text.strip()

def generate_secondary_kql(nlq: str, tables: List) -> str:
    
    table_schemas = []
    for table in tables:
        columns = ", ".join(table['columns'])
        table_schemas.append(f"{table['name']}: [{columns}]")
        
    prompt = f"""Generate a Kusto Query Language (KQL) query using these tables and columns and given an NLQ query explaining it:\n\nTables: {chr(10).join(table_schemas)}\nNLQ: {nlq}\n\nFollow these rules:\n1. Use proper joins (if multiple tables) using existing columns\n2. Include security-relevant filters using available columns\n3. Use appropriate aggregation functions from the schema\n4. Reference only existing columns from the tables\n5. Follow KQL best practices for performance\n\nReturn only the KQL query without comments"""
    
    response = client.models.generate_content(model = "gemini-2.0-flash", contents = prompt)
    return response.text.strip()