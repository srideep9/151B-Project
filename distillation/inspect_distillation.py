import json

# Replace with your actual 32B output file name
OUTPUT_FILE = "deepseek_r1_qwen32b_public_n31.jsonl"

with open(OUTPUT_FILE, 'r') as f:
    outputs = [json.loads(line) for line in f]

failed_count = 0
ids = []
for out in outputs:
    # Filter for Free-Response Questions that failed the Judger
    if not out.get("is_mcq") and out.get("correct") is False:
        q_id = out.get("id")
        gold = out.get("gold")
        response = out.get("response", "")
        
        print(f"\n" + "="*40)
        print(f"--- ID: {q_id} ---")
        print(f"GROUND TRUTH: {gold}")
        print("-" * 40)
        # Print the last 300 characters to catch the final box and any conversational filler
        print(f"MODEL RESPONSE (Tail End):\n{response[-300:]}") 
        
        failed_count += 1
        ids.append(q_id)
print(f"\nTotal Failed Free-Response Questions: {failed_count}")
print(f"Failed Question IDs: {ids}")