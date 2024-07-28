import json
import os
import sys

def fun1():
    """
    make a prediction file of origin set.
    for SQL setting, q is the same. golden sql is different.

    cd cache
    ln -s 
    """
    data1 = json.load(open("datasets/financial.json", "r"))
    qid_sql_map = {d["question"]: d["SQL"] for d in data1}
    for model_name in ["zero_shot", "din_sql"]:
        for usek in ["no_evidence_False", "no_evidence_True"]:
            # out/din_sql_FinancialCorrected_gpt-4o_no_evidence_False.json
            pin = f"out/{model_name}_FinancialCorrectedSQL_gpt-4o_{usek}.json"
            data = json.load(open(pin, "r"))
            for i in range(len(data)):
                q = data[i]["Question"]
                data[i]["Gold Query"] = qid_sql_map[q]
                del data[i]["Success"]
            pout = f"out/{model_name}_FinancialCorrected-orignal_gpt-4o_{usek}.json"
            # cd cache. ln -s {model_name}_FinancialCorrectedSQL_gpt-4o_{usek} {model_name}_FinancialCorrected-orignal_gpt-4o_{usek}
            os.system(f"cd cache; ln -s {model_name}_FinancialCorrectedSQL_gpt-4o_{usek} {model_name}_FinancialCorrected-orignal_gpt-4o_{usek}")
            json.dump(data, open(pout, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

from run_model import run_model_on_dataset

def print_results():
    """
    print the results of the prediction.
    Original		Corrected SQL		Corrected Data
    w/ Oracle Knowledge
        Zero-shot (GPT-4o)
        DIN-SQL (GPT-4o)
    w/o Oracle Knowledge
        Zero-shot (GPT-4o)
        DIN-SQL (GPT-4o)
    """
    llm_name = "gpt-4o"
    lines = []
    for no_evidence in [False, True]:
        lines.append(f"{'w/' if not no_evidence else 'w/o'} Oracle Knowledge")
        for model_name in ["zero_shot", "din_sql"]:
            line = model_name
            for dataset_name in ["FinancialCorrected-orignal", "FinancialCorrectedSQL", "FinancialCorrected"]:
                acc = run_model_on_dataset(model_name, dataset_name, llm_name, no_evidence,eval_mode=True)
                line += f"\t{acc:.4f}"
            lines.append(line)
    print("\n".join(lines))

if __name__ == "__main__":
    # python my.py
    # fun1()
    print_results()
