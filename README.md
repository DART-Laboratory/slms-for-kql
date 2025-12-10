# SLM Threat Query Analysis

The following Github Repository provides the codebase used to analyze Small Language Model (SLM) generation of Kusto Query Language (KQL) queries. This repository contains multiple experiments, including exploratory experiments of LLM-generated KQL queries from four LLMs: Google's Gemini 2.0 Flash, Microsoft's Phi-4, and OpenAI's GPT-4o and GPT-5, as well as six other SLMs: Google's Gemma-3-1B-IT and Gemma-3-4B-IT, Microsoft's Phi-4-Mini-Instruct, and Qwen 2.5-7B-Instruct-1M. We assess the syntax and semantics of these LLM-generated queries with the help of Microsoft's [KQL parser](https://github.com/microsoft/Kusto-Query-Language), and develop methodologies to improve the syntactic and semantic quality of KQL queries. Furthermore, we incorporate metrics that are outlined in [NL2KQL](https://arxiv.org/pdf/2404.02933). NL2KQL is a system that has been developed by Microsoft to develop KQL queries based on Natural Language Queries (NLQs). Although we perform experiments with LLMs as well, the primary focus of this research is on SLM effectiveness in KQL code generation.

### Background:

Kusto Query Language (KQL) is a Microsoft-based programming language that can utilized to analyze and decipher patterns within data. KQL is particularly useful in Microsoft Security logging products, such as Microsoft Defender 365/XDR for Advanced Hunting, Microsoft Sentinel, and Microsoft Azure. For more information on how KQL is utilized, see [here](https://learn.microsoft.com/en-us/kusto/query/?view=microsoft-fabric): In order to construct a basic KQL query, the syntax is as follows:
<br></br>
```
DataSource
| filters OR modifiers OR limiters
```
```DataSource``` specifies a data table of information to navigate through. Examples of commonly used DataSources in Advanced Hunting include ```DeviceProcessEvents```, ```DeviceLogonEvents```, and ```CloudAppEvents```, which keep track of data related to process creation, device logon information, and cloud accounts/objects respectively. 
<br></br>
```filters OR modifiers OR limiters``` serve as conditions that may be applied to results in order to receive a subset of the original data table. Common filtering commands for a ```DataSource``` include: ```take```, ```top```, ```project```, and ```where```.

### Problem Statement:

The advent of LLMs has resulted in greater feasibility in boilerplate code generation, and has proven to be at least somewhat helpful for programmers. However there is little guarantee that boilerplate code, irrespective of programming language, is always correct. The purpose of this project is to identify SLM/LLM shortcomings in KQL code generation, particularly in terms of syntax and semantics, and address new ways to rectify these shortcomings while minimizing computational costs, memory overhead, and reduce potential attack surfaces to preserve data security via SLMs. 

### Methodology:

There are multiple folders involved in the ```code/``` folder of this Github Repository. Below are descriptions about what each of the folders contain:
- ```Benchmarking_Pipeline``` - Pipeline that automates LLM and SLM model code generation in the Zero-Shot Prompting Scenario
- ```Fine_Tuned_Solutions``` - Contains outlines that were used to fine-tune SLMs. We test out both Supervised Fine-Tuning and Chain-of-Thought (CoT) Fine-Tuning with Low Rank Adaptation (LoRA).
- ```NL2KQL_Remakes``` - Contains our version of NL2KQL that we remade based on the specifications provided in the original [NL2KQL](https://arxiv.org/pdf/2404.02933) paper.
- ```Query_Refiner``` - Part of the NL2KQL pipeline that is used as an "extra step" to fix errors that may persist in LLM-generated KQL outputs. Using this feature involves .NET, and a runner python file of the query refiner is integrated within the files in ```NL2KQL_Remakes```.
- ```offline_metrics_pipeline/offline-metrics-pipeline``` - Pipeline that calculates different metric scores as specified in the original [NL2KQL](https://arxiv.org/pdf/2404.02933) paper. We use these metrics to quantify how effective and accurate the LLM-generated KQL queries are.
-----
### System Requirements:

All experiments for this paper were run using a machine with an AMD EPYC 9534 64-Core Processor, an NVIDIA H100 NVL GPU, and Ubuntu 22.04.5 LTS. Please ensure that similar systems are used within your experiments, as this can affect latency measures heavily depending on which GPUs/Core Processors are utilized. Furthermore, this can affect the amount of time it takes to train a model using LoRA.

-----

### Replication Steps:

#### Step #1: Package Installation and Virtual Environment (venv) Creation

In order to run the notebooks, you will need to setup a virtual environment. The list of packages needed to run the code contained in this repository are listed under the ```requirements.txt``` file under the code folder. 

1. Create a virtual environment by running the command ```python3 -m venv venv```. Ensure that you have ```python3``` installed on your machine first prior to creating the virtual environment.
2. Activate the virtual environment by running the command ```source venv/bin/activate```.
3. To install all packages into your virtual environment, navigate to the ```code``` folder within this repository and run the following command: ```pip install -r requirements.txt```

#### Step #2: .NET Installation

To install .NET via the command line, follow these instructions:
1. Visit the following [link](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) and view the possible .NET SDK installers, and click ```dotnet-install-scripts```. It is recommended to use .NET 8, as this repository has been tested with .NET 8.
2. Click the link under "Bash (PGP Signature)"; this should download a .sh file. Store this in the directory of your choice.
3. Run ```chmod 777 dotnet-install.sh``` to change the modification level of the file; this allows you, as a user, to run the code.
4. Run ```./dotnet-install.sh``` from the current directory to finish .NET installation. Follow all instructions that proceed during the process.
5. After installation is complete, verify that .NET is installed by running the command ```dotnet --list-sdks```

#### Step #3: Setup Configuration Files

In order to query multiple APIs, you will need to use API Keys/Tokens. You will need to provide your API Keys/Tokens for **Google AI**,  **OpenAI**, and **Huggingface**. Follow these instructions:

1. Navigate to ```code/config.yaml``` within the Github repository. There should be token options: ```genai```, ```openai```, and ```huggingface```. ```genai``` should contain your Google API Key, ```openai``` should contain your OpenAI API Key, and ```huggingface``` should contain your Huggingface API Key.
2. If you have not already, generate API Keys/Tokens for [Google](https://ai.google.dev/gemini-api/docs/api-key), [OpenAI](https://platform.openai.com/docs/quickstart/create-and-export-an-api-key), and [Huggingface](https://huggingface.co/docs/hub/en/security-tokens).
   - **IMPORTANT:** If you are querying any Google model using the ```genai``` mode, it is **HIGHLY RECOMMENDED** to upgrade to Tier 1 pricing, as this will lead to minimal disruptions when running code; Free Tier members are usually query restricted depending on the model (as of 11/9/25, Gemini 2.0 Flash is limited up to 200 queries per day on Free Tier, and will produce errors if queried more than 200 times per day under the same API key).
4. Update the ```config.yaml``` file with the appropriate tokens. If you wish to test other LLMs beyond this list, you may update the ```config.yaml``` file accordingly.

#### Step #4: Replicating Processes for RQ1

To run the same process for RQ1, follow these instructions:
1. Navigate to ```code/Benchmarking_Pipeline``` within the Github repository. There should be multiple notebooks that start with "Zero-Shot" (i.e. ```zero-shot-gemma1b.ipynb```, ```zero-shot-gemma4b.ipynb```, etc.). Each individual follows the same Zero-Shot pipeline, but with different models. 
2. Follow the instructions that are noted in each notebook. If you run into issues, ensure that Steps #1, #2, and #3 are properly followed. If issues persist, please leave an issue on the Github repository.

#### Step #5: Replicating Processes for RQ2 and RQ3

To run the same process to obtain results for RQ2 and RQ3, follow these instructions:
1. Navigate to ```code/NL2KQL_Remakes``` within the Github repository. There are two notebooks that should be used: ```nl2kql_remake_generalized-api.ipynb``` and ```nl2kql_remake_generalized-huggingface.ipynb```
  - The ```nl2kql_remake_generalized-api.ipynb``` should be used for testing any OpenAI or Gemini models (or any other Google models that utilize Google's GenAI SDK). In our case, we use this notebook to obtain NL2KQL results for GPT-4o, GPT-5, and Google Gemini 2.0 Flash.
  - The ```nl2kql_remake_generalized-huggingface.ipynb``` should be used for testing any Huggingface models. In our case, we use this notebook to test Google's Gemma-3-1B-IT, Gemma-3-4B-IT, Microsoft's Phi-4-Mini-Instruct, Phi-4-Mini, and Qwen 2.5-7B-Instruct-1M.
  - There are two variables to take note of in these notebooks: ```prompt_template_path``` and ```value_placeholder```. ```prompt_template_path``` specifies the path to the prompt that you wish to test out with NL2KQL. You may change this path accordingly, or even create your own prompt in a .txt file and change the ```prompt_template_path``` to point towards that file. ```value_placeholder``` specifies whether Microsoft Defender Values should be taken into account with your prompt. In the initial NL2KQL prompt, they are taken into account (therefore ```value_placeholder``` is set to ```True```). However, in our revised prompting strategies, they are not taken into account. Therefore, if using our prompting strategies, ```value_placeholder``` should be set to ```False```.
2. Visit the following [link](https://myuva-my.sharepoint.com/:u:/g/personal/dqb5ty_virginia_edu/IQArBzkiPfA7Tbz-7FXff-aCAbypmYnH6k8nDQXSLvnx-2E?e=9yIwTr) to download the value embeddings needed to run NL2KQL. Once downloaded you will then need to place this download in ```code/NL2KQL_Remakes/data/embeddings```. 
3. The following rule is specific to RQ3 - you will need to change two variables in ```nl2kql_remake_generalized-huggingface.ipynb```: ```prompt_template_path``` and ```value_placeholder```. When testing different prompts (i.e. Alternative Prompt #1 and Alternative Prompt #2 as highlighted in the paper), you will need to change the ```prompt_template_path``` to point towards ```prompt_template_new_one.txt``` and ```prompt_template_new_two.txt``` respectively. Furthermore, you will need to set the ```value_placeholder``` variable to ```False```, as these alternative prompts do not utilize value information compared to the original NL2KQL configuration.
4. Follow the instructions that are noted in each notebook. If you run into issues, ensure that Steps #1, #2, and #3 are properly followed. If issues persist, please leave an issue on the Github repository.

#### Step #6: Replicating Processes for RQ4:

To run the same process to obtain results for RQ4, follow these instructions:
1. Navigate to ```code/Fine_Tuned_Solutions``` within the Github repository. You will need to run the instructions that are present in ```gemma-3-4b-finetune-notebook.ipynb```.
   - Note that this notebook can be used for Supervised Fine-Tuning of the Gemma-3-4B-IT model, as well as the Chain-of-Thought (CoT) Fine-Tuning of the Gemma-3-4B-IT model.
   - The training data that was used for both Supervised Fine-Tuning and CoT Fine-Tuning is contained in ```training_data.csv``` within ```code/Fine_Tuned_Solutions```
2. Follow the instructions that are noted in the notebook. You can change whether to perform CoT Fine-Tuning or strict Supervised Fine-Tuning by changing the ```mode``` variable as noted in the Jupyter notebook. If you run into issues, ensure that Steps #1, #2, and #3 are properly followed. If issues persist, please leave an issue on the Github repository.
3. When running this notebook, multiple folders with model weights will be outputted. Furthermore, the end of the Jupyter notebook should give you the "best" performing model based off of evaluation/validation loss. You will now need to feed this model into the ```nl2kql_remake_generalized-huggingface.ipynb``` notebook, and update variables (i.e. ```mode```, ```prompt_template_path```, etc.) accordingly.

#### Step #7: Replicating Processes for RQ5 and RQ6:

To run the same process to obtain results for RQ5 and RQ6, follow these instructions:
1. Navigate to ```code/NL2KQL_Remakes``` within the Github repository. Find the notebook titled ```nl2kql_remake_generalized-two-stage-huggingface.ipynb```.
   - This notebook currently loads a SLM Huggingface model, Google's Gemma-3-4B-IT, as well as an Oracle LLM, Google's Gemini 2.0 Flash. If you wish to change the SLM model that is currently used in this pipeline, you may do so (see Step #5 under the pipeline).
2. Follow the instructions that are noted in the notebook. If you run into issues, ensure that Steps #1, #2, and #3 are properly followed. If issues persist, please leave an issue on the Github repository.
3. The following step applies to RQ6 - in order to replicate the results obtained from RQ6, you **will need** to change the evaluation dataset that is used. This data can be found in ```code/NL2KQL_Remakes/data/evaluation/sentinel_defender.csv```. This .csv file provides 83 NLQ-KQL pairs that were scraped from open source repositories. You will need to make the following changes to the ```nl2kql_remake_generalized-two-stage-huggingface.ipynb``` notebook:
   - Change the ```eval_df``` variable under "Step #5: Creating the Entire Pipeline" to point towards the .csv file.
   - Change the ```queries``` variable under "Step #5: Creating the Entire Pipeline" to ```list(eval_df['prompt'])``` rather than ```list(eval_df['context'])``` by default.
