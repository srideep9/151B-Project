from judger import Judger

# Initialize the judger
j = Judger()

print("--- Test 1: The Polynomial Zeros ---")
# Simulating the exact evaluation pass
pred_1 = "The answer is \\boxed{(-8, \\infty)}."
gold_1 = ['(-8,infinity)']
# auto_judge parameters: pred, gold, options
result_1 = j.auto_judge(pred=pred_1, gold=gold_1, options=[[]])
print(f"Polynomial Result: {result_1}\n")

print("--- Test 2: The Fraction vs Decimal ---")
pred_2 = "The length is \\boxed{\\dfrac{504}{11(3.1416) }}"
gold_2 = ["14.5843461351483"]
result_2 = j.auto_judge(pred=pred_2, gold=gold_2, options=[[]])
print(f"Pi Fraction Result: {result_2}\n")

print("--- Test 3: The Rounding Penalty ---")
pred_3 = "The answer is \\boxed{atan(4.76), \\pi}."
gold_3 = ['atan(4.76)', 'pi']
result_3 = j.auto_judge(pred=pred_3, gold=gold_3, options=[[]])
print(f"Rounding Result: {result_3}")