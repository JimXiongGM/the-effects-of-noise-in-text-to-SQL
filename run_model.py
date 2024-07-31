import argparse
import os,re
os.environ["http_proxy"] = "http://localhost:7893"
os.environ["https_proxy"] = "http://localhost:7893"

import logging 
# import datetime
import json

from src.models.zero_shot import ZeroShotModel
from src.models.din_sql import DinSQLModel
from src.datasets import get_dataset
from langchain.chat_models import ChatOpenAI
from tqdm import trange



# Load OpenAI API Key
api_key = os.environ.get('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Enable logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format=log_format)

# Suppress debug logs from OpenAI and requests libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def extract_sql_query(input_text):
    r1 = re.findall(r"```sql\n(.*?)\n```", input_text, re.DOTALL | re.IGNORECASE)
    r = r1[-1] if r1 else "None"
    return r.strip()

def run_model_on_dataset(model_name, dataset_name, llm_name, no_evidence,eval_mode=False):
    
    dataset = get_dataset(dataset_name)

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=llm_name,
        temperature=0,
        request_timeout=60
    )

    if model_name == "zero_shot":
        model = ZeroShotModel(llm)
    elif model_name == "din_sql":
        model = DinSQLModel(llm)        
    else: 
        raise ValueError("Supplied model_name not implemented")

    no_data_points = dataset.get_number_of_data_points()
    score = 0
    results = []

    # if model_name == "zero_shot":
    #     sql_schema = dataset.get_create_table_statements()
    #     # predicted_sql = model.generate_query(sql_schema, question, evidence)
    # elif model_name == "din_sql":
    #     sql_schema = dataset.get_schema_and_sample_data(db_id)
        
    #     # NOTE: 原作没有提供这个函数
    #     bird_table_info = dataset.get_bird_db_info(db_id)                
    # sql_schema = dataset.get_create_table_statements()
    for i in range(no_data_points):
        # debug
        # if i >= 3: break

        data_point = dataset.get_data_point(i)
        
        # evidence = data_point['evidence'] if no_evidence else ""
        evidence = "" if no_evidence else data_point['evidence']

        golden_sql = data_point['SQL']
        db_id = data_point['db_id']
        question = data_point['question']
        difficulty = data_point.get('difficulty', "")
        sql_schema = dataset.get_create_table_statements()

        # cache
        cache_dir = f"cache/{model_name}_{dataset_name}_{llm_name}_no_evidence_{no_evidence}"
        os.makedirs(cache_dir, exist_ok=True)
        question_id = data_point['question_id']
        cache_path = f"{cache_dir}/{question_id}.json"

        if eval_mode:
            assert os.path.exists(cache_path), f"Cache file {cache_path} does not exist"

        if os.path.exists(cache_path):
            predicted_sql = json.load(open(cache_path, 'r'))
        else:
            if model_name == "zero_shot":  
                predicted_sql = model.generate_query(sql_schema, question, evidence)
                predicted_sql = extract_sql_query(predicted_sql)
            elif model_name == "din_sql":
                bird_table_info = dataset.get_bird_table_info()
                predicted_sql = model.generate_query(sql_schema, bird_table_info, evidence, question)
            with open(cache_path, 'w') as file:
                json.dump(predicted_sql, file, ensure_ascii=False)

        success = dataset.execute_queries_and_match_data(predicted_sql, golden_sql)

        score += success
        accuracy = 100*score / (i + 1)

        results.append({
            "Question": question,
            "Gold Query": golden_sql,
            "Predicted Query": predicted_sql,
            "Success": success,
            "Difficulty": difficulty
        })

        print(f"Percentage done: {round(i / no_data_points * 100, 2)}% Domain: {db_id} Success: {success} Accuracy: {accuracy}")

    if not eval_mode:
        # Save results to JSON file
        # logs_dir = "logs"
        # os.makedirs(logs_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filepath = os.path.join(logs_dir, f"results_{timestamp}.json")
        os.makedirs("out", exist_ok=True)
        save_path = f"out/{model_name}_{dataset_name}_{llm_name}_no_evidence_{no_evidence}.json"

        with open(save_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"Saved results to {save_path}")
    print(f"Total Accuracy: {accuracy}")
    return accuracy



def run_model_statistics(model_name, dataset_name, llm_name, no_evidence,eval_mode=False):
    import tiktoken
    encoder = tiktoken.encoding_for_model("gpt-4o")
    def get_tokens_len(text):
        return len(encoder.encode(text))
    
    dataset = get_dataset(dataset_name)
    no_data_points = dataset.get_number_of_data_points()

    # add for statistics
    all_lens_sql_schemas = []
    all_lens_bird_table_info = []
    db_len_map = {}
    for i in trange(no_data_points,ncols=100,colour='green',desc=f"Running {model_name} on {dataset_name}"):
        data_point = dataset.get_data_point(i)
        db_id = data_point['db_id']
        if db_id not in db_len_map:
            sql_schema = dataset.get_create_table_statements()
            bird_table_info = dataset.get_bird_table_info()
            db_len_map[db_id] = {
                "sql_schema": get_tokens_len(sql_schema),
                "bird_table_info": get_tokens_len(bird_table_info)
            }

        all_lens_sql_schemas.append(db_len_map[db_id]["sql_schema"])
        all_lens_bird_table_info.append(db_len_map[db_id]["sql_schema"] + db_len_map[db_id]["bird_table_info"])
    
    # calculate statistics
    ave_sql_schema = sum(all_lens_sql_schemas) / len(all_lens_sql_schemas)
    ave_bird_table_info = sum(all_lens_bird_table_info) / len(all_lens_bird_table_info)
    print(f"Average SQL Schema Length: {ave_sql_schema}")
    print(f"Average Bird Table Info Length: {ave_bird_table_info}")

def main():
    parser = argparse.ArgumentParser(description='Run text-to-SQL models on specified datasets with an option to specify the OpenAI LLM to use.')

    # Set default values for model, dataset, and llm (language model) arguments
    parser.add_argument('--model', type=str, default='default_model_name', help='The name of the model to use (default: default_model_name)')
    parser.add_argument('--dataset', type=str, default='default_dataset_name', help='The name of the dataset to use (default: default_dataset_name)')
    parser.add_argument('--llm', type=str, default='GPT-3.5-Turbo', help='The OpenAI language model to use (default: GPT-3.5-Turbo)')
    parser.add_argument("--no_evidence", action="store_true", help="Do not add evidence to the question")

    args = parser.parse_args()

    print(f"No evidence: {args.no_evidence}")
    # exit()
    # Run the specified model on the specified dataset using the specified LLM
    # run_model_on_dataset(args.model, args.dataset, args.llm, args.no_evidence)
    
    # my add
    run_model_statistics(args.model, args.dataset, args.llm, args.no_evidence)

if __name__ == '__main__':
    main()
