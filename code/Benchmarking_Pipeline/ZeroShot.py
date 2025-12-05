import re
import yaml
import time
import pandas as pd

# For any GPT models:
from openai import OpenAI

import torch
import transformers
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Gemma3ForConditionalGeneration
from transformers import BitsAndBytesConfig
from huggingface_hub import login

# For Gemini-2.0 Flash:
from google import genai
from google.api_core import retry
from google.genai.types import GenerateContentConfig, HttpOptions

class ZeroShot:
    def __init__(self, model_name: str, mode: str, eval_df: pd.DataFrame):
        self.input_tokens = 0
        self.output_tokens = 0
        self.counter = 0
        self.times = []
        
        self.model_name = model_name
        self.mode = mode
        self.eval_df = eval_df
        self.df = pd.DataFrame(columns = ['NLQ', 'baseline', 'KQL', 'Full Response'])

        with open('../config.yaml', 'r') as f:
            keys = yaml.safe_load(f)

        if mode == 'huggingface':

            model_args = {
                "token": keys['huggingface']['token'],
                "dtype": torch.float32,
                "device_map": "auto",
            }
        
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                  **model_args).to("cuda")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                               token=keys['huggingface']['token'])
                
                # Pipeline configuration
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=model_args["dtype"],
                )
                
            except Exception as e:
                raise RuntimeError(f"Initialization failed: {str(e)}")
                
        elif mode == 'openai':
            self.client = OpenAI(api_key = keys['openai']['token'])
        elif mode == 'genai-client':
            self.client = genai.Client(api_key = keys['genai']['token'])
        else:
            self.client = None
            raise Exception("Mode Invalid: Please double check the mode to proceed.")

    def generate(self, connector: str) -> str:

        if self.mode == 'huggingface':

            messages = [
                {"role": "system", "content": "You are a programmer using the Kusto Query Language with Microsoft Defender. Generate a KQL query that answers the following request. Return only the KQL code without any explanation"},
                {"role": "user", "content": f"{self.eval_df.loc[self.counter]['context']}"},
            ]
            
            final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                              add_generation_prompt=True,
                                                              return_tensors="pt")

            input_tokens = self.tokenizer(self.tokenizer.decode(final_prompt[0]))
            self.input_tokens = self.input_tokens + len(input_tokens['input_ids'])
        
            start = time.time()
            full_response = self.pipeline(self.tokenizer.decode(final_prompt[0]), return_full_text=False)
            end = time.time()

            self.output_tokens = self.output_tokens + len(self.tokenizer(full_response[0]['generated_text'])["input_ids"])

            # Add the latency
            self.times.append(end-start)

            try:
                clean_response = re.search(r'```(?:kusto|kql)(.*)```', full_response[0]["generated_text"], flags=re.DOTALL).group(1)
            except:
                clean_response = full_response[0]["generated_text"].replace("`", "")
            
            print(clean_response)

        elif self.mode == 'genai-client':
            start = time.time()
            full_response = self.client.models.generate_content(model = self.model_name,
                                                           contents = f"{self.eval_df.loc[self.counter]['context']}",
                                                           config = GenerateContentConfig(system_instruction=[f'You are a programmer using the Kusto Query Language with Microsoft Defender. Generate a KQL query that answers the following request. Return only the KQL code without any explanation.']))
            end = time.time()

            # Add the latency
            self.times.append(end-start)
            
            try:
                clean_response = re.search(r'```(?:kusto|kql)(.*)```', full_response.text, flags=re.DOTALL).group(1)
            except:
                clean_response = full_response.text
                
            print(clean_response)
            
        elif self.mode == 'openai':
            start = time.time()
            full_response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": "You are a programmer using the Kusto Query Language with Microsoft Defender. Generate a KQL query that answers the following request.  Return only the KQL code without any explanation."},
                    {"role": "user", "content": f"{self.eval_df.loc[self.counter]['context']}"}
                ],
            )
            end = time.time()

            # Add the latency
            self.times.append(end-start)
            
            full_response = full_response.choices[0].message.content

            try:
                clean_response = re.search(r'```(?:kusto|kql)(.*)```', full_response, flags=re.DOTALL).group(1)
                print(clean_response)
            except:
                clean_response = full_response
                print(clean_response)

        self.df.loc[len(self.df)] = [self.eval_df.loc[self.counter]['context'], self.eval_df.loc[self.counter]['baseline'], clean_response, full_response]
        
        self.counter = self.counter + 1
        
        if self.mode == 'openai':
            time.sleep(20)
        elif self.mode == 'genai-client':
            time.sleep(5)
        
        return clean_response