import yaml
from typing import List, Dict, Tuple
import os
class QueryDataset:
    def __init__(self, yaml_file: str):
        self.queries = self._load_queries(yaml_file)
    
    def _load_queries(self, yaml_file: str) -> List[Dict]:
        """Load queries from YAML file"""
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data['queries']
    
    def get_query_by_id(self, query_id: int) -> Dict:
        """Get a specific query by ID"""
        return next((q for q in self.queries if q['id'] == query_id), None)
    
    def get_queries_by_connector(self, connector: str) -> List[Dict]:
        """Get all queries for a specific connector"""
        return [q for q in self.queries if connector in q['connector']]
    
    def get_all_queries(self) -> List[Dict]:
        """Get all queries"""
        return [q for q in self.queries]
    
    def get_query_details(self, query: Dict) -> Tuple:
        query_id = query['id']
        prompt = query['prompt']
        connector = query['connector']
        baseline = query['baseline']
        llmResult = query['llmResult']
        
        return query_id, prompt, connector, baseline, llmResult
    
    def display_query_details(self, query: Dict) -> None:
        """Print formatted query details"""
        print(f"\nQuery ID: {query['id']}")
        print(f"Prompt: {query['prompt']}")
        print(f"Connector: {query['connector']}")
        print(f"KQL Query:{query['baseline']}")
        print(f"Column Definition:{query['llmResult']}")