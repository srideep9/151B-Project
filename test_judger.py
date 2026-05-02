from judger import Judger

# Initialize the judger
j = Judger(strict_extract=False)

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

print("--- Test 4: The Rounding Penalty ---")
pred_3 = """Okay, let's tackle this problem step by step. First, the question is asking for a point in the third quadrant where tan(α) = 3. Then, for that angle, find sin(α). 
First, I need to recall that in the third quadrant, both sine and cosine are negative. The tangent is positive there because it's sine over cosine, both negative, so their ratio is positive. 

Tan(α) is 3, which is y/x for a point (x, y) on the terminal side of angle α. So, if tan(α) = y/x = 3, then y = 3x. Since it's in the third quadrant, both x and y should be negative. So let's pick a value for x. Let's say x = -1 (since we can choose any x, but to make it simple, maybe x = -1). Then y = 3x = 3*(-1) = -3. So the point would be (-1, -3). Wait, but let me check if that's correct. 

Wait, tan(α) is y/x, so if the point is (x, y), then tan(α) = y/x. So if we want tan(α) = 3, then y/x = 3, so y = 3x. In the third quadrant, x < 0 and y < 0, so x is negative, y is 3 times x, so also negative. So a point could be (-1, -3), (-2, -6), etc. The problem says "determine any point", so we can pick one. Let's go with (-1, -3) as a valid point.

Now, for that angle, sin(α) is y/r, where r is the radius (distance from origin to the point). So r = sqrt(x² + y²). For the point (-1, -3), r = sqrt((-1)² + (-3)²) = sqrt(1 + 9) = sqrt(10). So sin(α) = y/r = (-3)/sqrt(10). We can rationalize the denominator: (-3 sqrt(10))/10. But the problem says "exact form", so probably that's the answer.

Wait, let me confirm. Let's make sure. If the point is (x, y) = (-1, -3), then tan(α) = y/x = (-3)/(-1) = 3, which is correct. In the third quadrant, sin(α) is negative, which matches. So sin(α) = -3/sqrt(10) = -3√10/10.

The problem says "one valid point" so we need to write the point. Let's check if (-1, -3) is correct. Yes, because tan(α) = 3, and in third quadrant, both coordinates negative.

So the first answer is the point (-1, -3), and the second is sin(α) = -3√10/10.

Wait, but the problem says "Determine any point in the third quadrant for which tan(α)=3". So the point is (x, y), so we need to write it as (x, y). Let's make sure we pick a point where x and y are integers, maybe. (-1, -3) is a good choice.

So the first answer is (-1, -3), and the second is -3√10/10.

Let me check the exact value. For sin(α), it's y/r, so y is -3, r is sqrt(1 + 9) = sqrt(10), so sin(α) = -3/sqrt(10) = -3√10/10. That's the exact form.

The problem says "exact form", so we need to present it as -3√10/10.

So the answers are: the point is (-1, -3), and sin(α) is -3√10/10.

Wait, the problem says "One valid point=[ANS] For this angle, sin(α)=[ANS]". So two answers. The first is the point, the second is sin(α).

So the first answer is (-1, -3), the second is (-3√10)/10.

Let me check if there's another point. For example, (-2, -6) would also work, but the problem says "any point", so (-1, -3) is fine.

So the boxed answers should be \boxed{(-1, -3)} and \boxed{-\dfrac{3\sqrt{10}}{10}}.

Wait, but the problem says "If the problem has multiple sub-answers, separate them by commas inside a single box". Wait, the user wrote: "Determine any point in the third quadrant for which tan(α)=3 One valid point=[ANS] For this angle, sin(α)=[ANS]"

So it's two sub-answers. So the answer should be two boxed answers? Wait, the problem says "put your final answer inside \boxed{}". But the user says "If the problem has multiple sub-answers, separate them by commas inside a single box".

Wait, the user's instruction says: "If the problem has multiple sub-answers, separate them by commas inside a single box, e.g., \boxed{3, 7}."

So here, the problem has two sub-answers: the point and sin(α). So the answer is \boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}.

But let's check the formatting. The problem says "NO UNITS" and "exact form". The point is written with coordinates, so (-1, -3) is correct. The sin(α) is -3√10/10.

Wait, but in the problem statement, the user wrote "One valid point=[ANS] For this angle, sin(α)=[ANS]". So two answers. So the answer should be two parts. The user says "put your final answer inside \boxed{}". So since there are two answers, we need to put them in a single box separated by commas.

Let me confirm the point. Let's take x = -1, then y = 3x = -3. So the point is (-1, -3). Correct.

Sin(α) is y/r = -3 / sqrt(1 + 9) = -3 / sqrt(10) = -3√10 / 10. Correct.

So the answers are (-1, -3) and -3√10/10.

So the boxed answer is \boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}.

Wait, but the problem says "CRITICAL FORMATTING RULES: 3. COORDINATES & INTERVALS: Always include the enclosing parentheses or brackets, e.g., \boxed{(2, -2)}."

So the point is written as (x, y), so (-1, -3) is correct.

Also, the sin(α) is a single value, so the second part is -3√10/10.

Let me check if the problem wants the point as a coordinate pair. Yes, the problem says "determine any point", so the answer is a point (x, y), so in coordinate form.

So the final answer is two parts: the point and sin(α). So the box should have both.

Yes, the user says "If the problem has multiple sub-answers, separate them by commas inside a single box".

So the answer is \boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}.

Wait, but in the example, they have \boxed{3, 7}, so the commas are inside the box. So the first part is the point, the second is the sine value.

Let me check if the problem says "One valid point=[ANS] For this angle, sin(α)=[ANS]". So two answers. So the answer is two values, so the box has both.

Yes.

So the answer is \boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}.

But let me make sure that the point is correctly written. In the third quadrant, x and y are negative, so (-1, -3) is correct.

Another check: tan(α) = y/x = (-3)/(-1) = 3, which is correct.

Sin(α) = y/r = -3 / sqrt(10) = -3√10 / 10. Correct.

Yes, that's right.

So the final answer is as above.
</think>

To determine a point in the third quadrant where $\tan(\alpha) = 3$, we begin by recalling that for an angle $\alpha$ in standard position, the tangent is defined as:

$$
\tan(\alpha) = \frac{y}{x}
$$

Given $\tan(\alpha) = 3$, this implies $y = 3x$. Since the point lies in the **third quadrant**, both $x < 0$ and $y < 0$. Choosing $x = -1$, we get $y = -3$, so the point is:

$$
(-1, -3)
$$

Next, to find $\sin(\alpha)$ for this angle, we use the definition of sine in terms of coordinates:

$$
\sin(\alpha) = \frac{y}{r}, \quad \text{where } r = \sqrt{x^2 + y^2}
$$

Substituting $x = -1$, $y = -3$:

$$
r = \sqrt{(-1)^2 + (-3)^2} = \sqrt{1 + 9} = \sqrt{10}
$$

$$
\sin(\alpha) = \frac{-3}{\sqrt{10}} = -\frac{3\sqrt{10}}{10}
$$

This is the exact value of $\sin(\alpha)$.

---

The problem asks for **one valid point** and the corresponding **$\sin(\alpha)$**. As per the instructions, multiple answers should be separated by commas inside a single box.

$$
\\boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}
$$
"""
pred_3_1 = "\\boxed{(-1, -3), -\dfrac{3\sqrt{10}}{10}}"
gold_3 = ['(-1,-3)', '-0.948683298050514']
result_3 = j.auto_judge(pred=pred_3, gold=gold_3, options=[[]])
print(f"Rounding Result: {result_3}")

print("--- Test 5---")
pred_4 = """
Okay, let's see. The problem is to express 2^{-5} + 5^{-1} as a reduced fraction. Hmm, first I need to remember that negative exponents mean reciprocals. So 2^{-5} is 1/(2^5) and 5^{-1} is 1/5. Let me write that down.

First, calculate each term. 2^5 is 32, so 2^{-5} is 1/32. Then 5^{-1} is 1/5. So the sum is 1/32 + 1/5. To add these fractions, I need a common denominator. The denominators are 32 and 5, which are coprime, so the least common denominator (LCD) is 32*5=160.

So convert 1/32 to 5/160 (since 32*5=160, multiply numerator and denominator by 5) and 1/5 to 32/160 (since 5*32=160, multiply numerator and denominator by 32). Then adding them gives (5 + 32)/160 = 37/160.

Wait, let me check that. 1/32 is 5/160? Let's see: 160 divided by 32 is 5, so yes, 1/32 = 5/160. And 1/5 is 32/160 because 160 divided by 5 is 32. So 5 + 32 is 37, so 37/160. Is that reduced? Let's check if 37 and 160 have any common factors. 37 is a prime number (since it's not divisible by 2,3,5,7; 7*5=35, 7*6=42 which is more than 37). 160 factors into 2^5 * 5. 37 is prime and doesn't divide 160, so yes, 37/160 is reduced.

The problem says to answer as a reduced fraction, so the answer is 37/160. Let me make sure I didn't make a mistake in the addition. 5 + 32 is 37, over 160. Yep, that's right. So the reduced fraction is 37/160.
</think>

2^{-5} = \frac{1}{32}, 5^{-1} = \frac{1}{5}. Add fractions: $\frac{1}{32} + \frac{1}{5} = \frac{5}{160} + \frac{32}{160} = \frac{37}{160}$.

\\boxed{\dfrac{37}{160}}
"""
gold_4 = ['37/160']
result_4 = j.auto_judge(pred=pred_4, gold=gold_4, options=[[]])
print(f"Rounding Result: {result_4}")

# 45, 56