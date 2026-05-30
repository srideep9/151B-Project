from judger import Judger

# Initialize the judger
j = Judger(strict_extract=False)

print("--- Test 1 ---")
# Simulating the exact evaluation pass
pred_1 = r"$$\boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}$$"
gold_1 = ['(-1,-3)', '-0.948683298050514']
# auto_judge parameters: pred, gold, options
result_1 = j.auto_judge(pred=pred_1, gold=gold_1, options=[[]])
print(f"Result: {result_1}\n")