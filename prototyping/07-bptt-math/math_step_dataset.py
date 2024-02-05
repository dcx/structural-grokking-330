import csv
import random
import logging
from typing import Optional
from enum import Enum, auto
from dataclasses import dataclass, asdict

from tqdm import tqdm

import torch
from rich.logging import RichHandler


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


class Operator(Enum):
    ADD = auto()
    MUL = auto()


class StepWriter:
    def __init__(self, problem, op):
        self.final = []

        self.problem = problem
        self.op = op

        self.steps = []

    def write_step(self, operator, problem, carry, shifted_out, final_ans, piece=None):

        step = {}

        logger.info(f"{'*'*100}")
        logger.info(f"Operator: {operator}, Problem: {problem}, Result: {final_ans}")
        logger.info(piece)

        logger.info(f"{self.final = }")


        s1_view = self.problem + '|' + ''.join([str(i) for i in shifted_out])

        s2_view = carry

        final_out = None


        step['s1_view'] = s1_view
        step['s2_view'] = s2_view
        step['final_out'] = final_out





        logger.error(f"op: s1_view = {s1_view}, s2_view = {s2_view}, final_out = {final_out}")

        # 456*789|[[4, 3, 7, 4], [0, 5, 4, 9, 3], [0, 0, 6, 5, 1, 3]]
        s1_view = ...

        # Carry:  [[0, 5, 5, 4], [0, 0, 4, 4, 3], [0, 0, 0, 3, 3, 3]]
        # Occluding carry will allow for s2 implicit carry
        mem_view =  ...

        #           4                8                   7  9  5  3                     
        final_ans = ...



def int_to_list(integer, reverse=False) -> list:
    int_list = [int(i) for i in [*str(integer)]]
    return int_list[::-1] if reverse else int_list

def list_to_int(_list) -> int:
    return int(''.join(map(str, _list[::-1])))


def add(adder, addend):
    logger.info("\n")
    logger.info(f"Adder: {adder}, Addend: {addend}")
    problem = f"{adder}+{addend}"

    writer = StepWriter(problem=problem, op=Operator.ADD)


    adder = int_to_list(adder, reverse=True)
    addend = int_to_list(addend, reverse=True)
    
    length = max(len(adder), len(addend))
    adder += [0] * (length - len(adder))
    addend += [0] * (length - len(addend))
    
    result_list = []
    carry_list = []
    carry = 0
    for a, b in zip(adder, addend):

        carry_list.append(carry)

        total = a + b + carry
        result_list.append(total % 10)
        

        result = list_to_int(result_list)
        logger.debug(f"Adding {a} + {b} + carry {carry} = {total}, New carry = {carry}, Result so far = {result}")

        
        writer.write_step('+', problem, result_list, carry_list, result, piece=f"{a}+{b}+{carry}={total}")
        carry = total // 10

    if carry != 0:
        result_list.append(carry)
        carry_list.append(carry)


    return list_to_int(result_list)




def multiply(multiplier, multiplicand):
    logger.info("\n")
    logger.info(f"Multiplier: {multiplier}, Multiplicand: {multiplicand}")
    problem = f"{multiplier}*{multiplicand}"

    writer = StepWriter(problem=problem, op=Operator.MUL)

    multiplier = int_to_list(multiplier, reverse=True)
    multiplicand = int_to_list(multiplicand, reverse=True)
    result = 0

    carry_list = []
    answer_list = []


    carry = 0

    for i, mler in enumerate(multiplier):

        for j, mand in enumerate(multiplicand):
            carry_list.append(carry)
            scaling_factor = 10 ** (i + j)
            answer = mler * mand

            answer_list.append((answer + carry) % 10)

            carry = answer // 10
            scaled =  answer * scaling_factor

            writer.write_step('*', problem, answer_list, carry_list, result, piece=f"{mler}*{mand}={answer}")


            if scaled < 100 and result == 0:
                logger.debug(f"No need to add, result is {result}: set result to {scaled} ")
                result = answer

            else:
                logger.debug(f"Adding partial product: {scaled} to total {result}")
                # Add these to memory temporarily when using add.
                result = add(result, scaled)

            
            if i == len(multiplier) - 1:
                writer.final.append(result)
            else:
                if j == 0:
                    writer.final.append(result)

        if carry > 0:
            answer_list.append(carry)
            carry_list.append(carry)
            carry = 0
            
        carry_list.append('|')
        answer_list.append('|')

    return result, carry_list, answer_list


def test(min_digit, max_digit, num_tests=100):
    logger.setLevel(logging.ERROR)

    for _ in range(num_tests):

        er = random.randint(10**min_digit, 10**max_digit)
        nd = random.randint(10**min_digit, 10**max_digit)

        real_mul = er*nd
        emp_mul, _ = multiply(er, nd)

        assert real_mul == emp_mul, f"{real_mul = } != {emp_mul = }"

        real_add = er+nd
        emp_add, _ = add(er, nd)
        assert real_add == emp_add, f"{real_add = } != {emp_add = }"


class MathDataset(torch.utils.data.Dataset):
    """
    Toy dataset for computing simple multiplication and addition problems.

    """

    def __init__(self, n_examples, min_digit, max_digit):
        self.n_examples = n_examples
        self.min_digit = min_digit
        self.max_digit = max_digit

        self.pad_token_id = 15
        self.sep_token_id = 16

        self.char_map = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5':5, '6': 6, '7': 7, '8': 8, '9': 9,
            '*': 10, '+': 11, '=': 12, ' ': 13, ',': 14,
            '#': self.pad_token_id, # pad token 
            '|': self.sep_token_id, # separator token           
        }

        self.data = []
        increment = 0
        for idx in tqdm(range(n_examples)):
            er = random.randint(10**min_digit, 10**max_digit)
            nd = random.randint(10**min_digit, 10**max_digit)

            _, step_list = multiply(er, nd)


            for step_idx, step in enumerate(step_list):
                tokenized_problem = self.tokenize(step.problem)
                tokenized_memory = self.tokenize(step.memory_concat)

                step_data = [
                    increment, idx, step_idx, tokenized_problem, tokenized_memory,
                    step.operator, step.problem, step.digits_involved,
                    step.carry, step.intermediate_result, step.final_result,
                    step.add_memory, step.mul_memory, step.scaled_answer,
                    step.memory_concat, 
                ]
                increment += 1

                self.data.append(step_data)

                
    
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, idx):
        return self.data[idx]

    def tokenize(self, s):
        a = [self.char_map[c] for c in s]
        return a
    
    def decode(self, a):
        s = ''.join([list(self.char_map.keys())[i] for i in a])
        return s
        

def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        """
        Pads batch of variable length.
        """
        longest_seq = max([max(len(x),len(y)) for x,y in batch])
        n_items = len(batch)

        x_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id
        y_batch = torch.zeros((n_items, longest_seq), dtype=torch.long) + pad_token_id

        for i, (x, y) in enumerate(batch):
            x_batch[i, :len(x)] = torch.tensor(x)
            y_batch[i, :len(y)] = torch.tensor(y)

        return x_batch, y_batch, (y_batch == pad_token_id)

    return collate_fn


def make_datasets(n_train, n_val):
        ds_train = MathDataset(n_examples=n_train, min_digit=0, max_digit=2)
        ds_val = MathDataset(n_examples=n_val, min_digit=0, max_digit=2)

        return ds_train, ds_val


if __name__ == '__main__':



    out, answer_list, carry_list = multiply(456,789)


    print(out, answer_list, carry_list)

    # ds = MathDataset(n_examples=10, min_digit=0, max_digit=3)
    # print(ds.data)
