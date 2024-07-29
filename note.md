
两个 setting
一个是已经纠正了SQL的数据集
另一个是已经纠正了SQL查询和有噪声的问题的数据集


```bash
# has Evidence
python run_model.py --model zero_shot --dataset FinancialCorrectedSQL --llm gpt-4o
python run_model.py --model zero_shot --dataset FinancialCorrected --llm gpt-4o

python run_model.py --model din_sql --dataset FinancialCorrectedSQL --llm gpt-4o
python run_model.py --model din_sql --dataset FinancialCorrected --llm gpt-4o

# No Evidence
python run_model.py --model zero_shot --dataset FinancialCorrectedSQL --llm gpt-4o --no_evidence
python run_model.py --model zero_shot --dataset FinancialCorrected --llm gpt-4o --no_evidence

python run_model.py --model din_sql --dataset FinancialCorrectedSQL --llm gpt-4o --no_evidence
python run_model.py --model din_sql --dataset FinancialCorrected --llm gpt-4o --no_evidence


python run_model.py --model zero_shot --dataset minidev --llm gpt-4o --no_evidence
python run_model.py --model din_sql --dataset minidev --llm gpt-4o --no_evidence
```

统计信息
```bash
python run_model.py --model din_sql --dataset bird-dev --llm gpt-4o
python run_model.py --model din_sql --dataset spider-dev --llm gpt-4o
```