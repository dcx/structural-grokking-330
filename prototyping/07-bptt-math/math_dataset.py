import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Step:
    answer: str
    scratchpad: str
    carry: str
    internal_memory: str
    internal_steps: List[int]



def int_to_list(n, reverse=False):
    lst = [int(i) for i in str(n)]
    return lst[::-1] if reverse else lst

def long_multiplication(multiplicand, multiplier, no_final_carry=False, zero_extend=False):
    multiplicand_list = int_to_list(multiplicand, reverse=True)
    multiplier_list = int_to_list(multiplier, reverse=True)

    steps = []
    carry_steps = []


    for multiplier_idx, multiplier_digit in enumerate(multiplier_list):
        row_result = []
        row_carry = []
        carry = 0

        for multiplicand_idx, multiplicand_digit in enumerate(multiplicand_list):
            product = multiplicand_digit * multiplier_digit + carry
            row_result.append(product % 10)

            # Intermediate carry
            row_carry.append(carry)
            carry = product // 10
                

            if multiplicand_idx == len(multiplicand_list) - 1 and no_final_carry:
                # Remove last step for final carry
                row_result.append(product)
                carry = 0


    
        # Final carry for each row
        if carry > 0:
            row_carry.append(carry)
            row_result.append(carry)

        if zero_extend:
            if multiplier_idx > 0:
                row_result.extend([0]*multiplier_idx)
                row_carry.extend([0]*multiplier_idx)

        steps.append(row_result)
        carry_steps.append(row_carry)

        


    return steps, carry_steps

multiplicand = 789
multiplier = 456


result_steps, carry_steps = long_multiplication(multiplicand, multiplier, no_final_carry=False, zero_extend=True)

print(f"{result_steps = }")
print(f"{carry_steps = }")