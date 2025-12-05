import yaml
import json
from ZeroShot import ZeroShot

class KQLBenchmarkPipeline:
    def __init__(self, llm_client: ZeroShot):
        self.llm_client = llm_client
        self.data = []
        self.results = {"queries": []}

    def _load_data(self):
        """Load sentinel and defender eval files"""
        # load defender eval data
        for idx, row in self.llm_client.eval_df[['context', 'baseline']].iterrows():
            defender_data = [{'context': row['context'], 'baseline': row['baseline']}]
            for item in defender_data:
                item["connector"] = "Defender"
            self.data.extend(defender_data)

    def generate_kql(self, connector: str):
        """Generate KQL using the LLM Client"""
        return self.llm_client.generate(connector)

    def run(self):
        """Process evaluation data using LLM Client"""
        self._load_data()
        for idx, item in enumerate(self.data, start=1):
            connector = item["connector"]
            llm_result = self.generate_kql(connector)
            
            self.results["queries"].append({
                "id": idx,
                "prompt": item["context"],
                "connector": connector,
                "baseline": item["baseline"].strip(),
                "llmResult": llm_result
            })
    
    def save_results(self, output_file="results.yaml"):
        """Format and save output to YAML file"""
        with open(output_file, 'w') as f:
            yaml.dump(
                self.results, 
                f, 
                sort_keys=False,
                default_style='|',
                allow_unicode=True,
                width=1000
            )