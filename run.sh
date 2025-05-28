# python main.py --output_dir "final1" --verbose
nohup python main.py --model_name Qwen/Qwen2.5-1.5B-Instruct --temperature 0.7     --dataset_name gsm8k --evaluator gsm8k --output_dir "final_gsm8k_qwen15_t07_chain8_reason_len_reward" --num_chains 8 --verbose     > a.log &
python plotter.py --log_dir "final1"