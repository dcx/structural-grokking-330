import logging
import random
from enum import Enum, auto

import torch
from rich.logging import RichHandler


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


class Operator(Enum):
    ADD = auto()
    MUL = auto()
    SUB = auto()
    DIV = auto()
    MULCAR = auto()
    ADDCAR = auto()


def op_to_symbol(op):
    mapping = {
        Operator.ADD: "+",
        Operator.MUL: "*",
        Operator.SUB: "-",
        Operator.DIV: "/",
        Operator.MULCAR: "x",
        Operator.ADDCAR: "+",
    }
    return mapping[op]

class StepWriter:
    def __init__(self, problem=None, op=None):
        self.problem = problem
        self.op = op

        self.add_carry = []
        self.mul_carry = []
        
    def write_step(self, op=None, intermediate=None, carry=None, final_computable=None):


        add_carry = ""
        mul_carry = ""
        final_result = []
        if op == Operator.ADD or op == Operator.ADDCAR:
            self.add_carry.append(carry)
            add_carry = ''.join(str(i) for i in self.add_carry)
            intermediate_str =  ''.join([str(i) for i in intermediate])
            

        elif op == Operator.MUL or op == Operator.MULCAR:
            self.mul_carry.append(carry)
            mul_carry = ''.join(str(i) for i in self.mul_carry)
            intermediate_str =  "|".join([''.join(map(str, i)) for i in intermediate])

        if final_computable:
            print('final computable')
            # shifted = left_shift(intermediate)
            # final_result.append(add(shifted, log=False))

            
        if intermediate is not None:
            # 456*789|[[4, 3, 7, 4], [0, 5, 4, 9, 3], [0, 0, 6, 5, 1, 3]]
            if op == Operator.MUL or op == Operator.MULCAR:
                s1_view = self.problem + "|" + intermediate_str
                print(s1_view)




        s2_view = f"{op_to_symbol(op)}|{mul_carry}|{add_carry}"

        

        # print(s2_view) 

        # Carry:  [[0, 5, 5, 4], [0, 0, 4, 4, 3], [0, 0, 0, 3, 3, 3]]
        # Occluding carry will allow for s2 implicit carry


        #           4                8                   7  9  5  3


        
            
        


def int_to_list(integer, reverse=False) -> list:
    int_list = [int(i) for i in [*str(integer)]]
    return int_list[::-1] if reverse else int_list


def list_to_int(_list) -> int:
    return int("".join(map(str, _list[::-1])))


def left_shift(rows):
    # Extend zeros for each row to align left shift
    for idx, row in enumerate(rows):
        rows[idx] = [0] * idx + row

    return rows

def add(addition_rows, writer=None, log=True):
    if not writer:
        problem = '+'.join([str(list_to_int(i)) for i in addition_rows])
        writer = StepWriter(problem=problem, op=Operator.ADD)

    result = []
    carry = 0
    
    maximum_col = max(len(i) for i in addition_rows)

    for index in range(maximum_col):
        column = [lst[index] for lst in addition_rows if index < len(lst)]
        column_sum = sum(column) + carry
        result.append(column_sum % 10)
        carry = column_sum // 10

        if log:
            writer.write_step(op=Operator.ADD, intermediate=result, carry=carry)

    while carry > 0:
        result.append(carry % 10)
        if log:
            writer.write_step(op=Operator.ADDCAR, intermediate=result, carry=carry)
        carry //= 10

    return list_to_int(result)


def multiply(multiplier, multiplicand, writer=None):

    m1_digits = int_to_list(multiplier, reverse=True)
    m2_digits = int_to_list(multiplicand, reverse=True)

    result_rows = []
    carry = 0


    for i, m1 in enumerate(m1_digits):
        result_rows.append([])
        carry = 0
        for j, m2 in enumerate(m2_digits):
            out = m1 * m2
            if carry:
                c = int_to_list(carry)
                o = int_to_list(out, reverse=True)
                out = add([c, o])
            carry = out // 10
            result = out % 10
            result_rows[i].append(result)
            
            if j == 0:
                final_result = True


            writer.write_step(op=Operator.MUL, intermediate=result_rows, carry=carry, final_computable=final_result)


        if carry:
            result_rows[i].append(carry)
            writer.write_step(op=Operator.MULCAR, carry=carry, intermediate=result_rows)
        
            

    return result_rows


def full_multiply(multiplier, multiplicand):

    writer = StepWriter(problem=f"{multiplier}*{multiplicand}", op=Operator.MUL)

    result_rows = multiply(multiplier, multiplicand, writer=writer)
    shifted_rows = left_shift(result_rows)
    added_row = add(shifted_rows)
    return added_row


def test(floor, ceiling, num_tests=10):

    for _ in range(num_tests):
        m1 = random.randint(floor, ceiling)
        m2 = random.randint(floor, ceiling)
        addition_rows = multiply(m1, m2)
        shifted_rows = left_shift(addition_rows)
        added_row = add(shifted_rows)
        true_added = true_add(shifted_rows)
        assert added_row == true_added
        assert true_added == m1 * m2


def true_add(addition_matrix):
    numbers = [list_to_int(i) for i in addition_matrix]
    return sum(numbers)

# test(0, 1000000, num_tests=100)

full_multiply(456, 789)
