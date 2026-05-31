import json
import re
import csv

def extract_nested_boxes(text):
    """Safely extracts contents of \boxed{} even with nested LaTeX braces."""
    results = []
    import re
    # Find every starting index of \boxed{
    for match in re.finditer(r'\\boxed\{', text):
        start = match.end()
        brace_count = 1
        i = start
        # Walk forward and count braces until they balance to 0
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
            
        if brace_count == 0:
            # We found the matching closing brace
            results.append(text[start:i-1].strip())
    return results

def fix_easy_formatting(input_file="distillation/deepseek_r1_qwen32b_public_n31.jsonl", question_file="data/public.jsonl",
                        output_jsonl="distillation/fixed_formatting.jsonl", review_csv="distillation/manual_review.csv"):
    cleaned_data = []
    review_rows = []

    with open(question_file, 'r') as qf:
        questions = {
            obj["id"]: obj
            for obj in map(json.loads, qf)
        }

    
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            response = data.get("response", "")
            gold = data.get("gold")
            is_mcq = data.get("is_mcq")
            q_id = data.get("id")
            question = questions[q_id]["question"]
            correct = data.get("correct")
            
            if "</think>" in response:
                reasoning = response.split("</think>")[0].replace("<think>", "").strip()
                tail_text = response.split("</think>")[-1]
            else:
                reasoning = response.strip()
                tail_text = response
            
            # Use the new safe extractor instead of regex
            box_contents = extract_nested_boxes(tail_text)
            
            # If the model didn't use \boxed{} at all but originally got it right (like MCQs),
            # we should preserve the original response to avoid losing valid data.
            if not box_contents:
                if correct:
                    cleaned_data.append(data)
                print(q_id)
                continue 
                
            # Clean and merge boxes safely
            clean_contents = [item.strip() for item in box_contents]
            merged_box = f"\\boxed{{{', '.join(clean_contents)}}}"
            
            new_response = f"<think>\n{reasoning}\n</think>\n{merged_box}"
            
            cleaned_data.append({
                "id": q_id,
                "is_mcq": is_mcq,
                "gold": gold,
                "response": new_response
            })
            
            # Flag for manual review if it originally failed the judger and is a Free-Response Question
            if not correct and not is_mcq:
                review_rows.append({
                    "id": q_id,
                    "gold_answer": gold,
                    "model_extracted": clean_contents,
                    "question": question
                })

    # Output the structurally fixed JSONL
    with open(output_jsonl, 'w') as f:
        for row in cleaned_data:
            f.write(json.dumps(row) + "\n")
            
    # Output the manual review CSV
    if review_rows:
        keys = review_rows[0].keys()
        with open(review_csv, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(review_rows)

    print(f"Format fixing complete. Saved {len(cleaned_data)} cleaned rows.")
    print(f"Generated {review_csv} with {len(review_rows)} items for manual decimal review.")

if __name__ == "__main__":
    fix_easy_formatting()