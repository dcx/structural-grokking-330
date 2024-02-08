import logging
import random
from typing import Optional
from dataclasses import dataclass, fields, asdict
from enum import Enum, auto
import csv
import os
import tqdm

import torch
from rich.logging import RichHandler


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


@dataclass
class Result:
    problem: str
    final_result: int
    operator: str
    step: int
    system: int
    cumulative_result: int
    s1_result: str
    s1_scratchpad: str
    cur_prob: str
    s2_carry: str
    s2_carry_scratchpad: str


    # Collated data
    # c_problem: Optional[str] = None
    # c_s1_result: Optional[str] = None
    # c_s1_scratchpad: Optional[str] = None
    # c_s2_carry: Optional[str] = None
    # c_s2_carry_scratchpad: Optional[str] = None
    # c_cumulative_result: Optional[int] = None
    # c_final_result: Optional[int] = None

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
        Operator.ADDCAR: "+",
        Operator.MULCAR: "*",
    }
    return mapping[op]


class StepWriter:
    def __init__(self, problem='', op=None, final_result=None, only_mul=True):
        self.problem = problem
        self.op = op

        self.add_carry = []
        self.mul_carry = []

        self.final_result = final_result

        self.s1_result_only = None
        self.s1_scratchpad = None
        self.s2_carry = None
        self.s2_carry_scratchpad = None
        self.intermediate_str = None
        self.only_mul = True

        self.results = []
        self.step_counter = 0

    def write_step(self, cur_prob=None, op=None, intermediate=None, carry=None, cumulative_result=[], primary=True):

        add_carry = ""
        mul_carry = ""
        system = 1

        cumulative_result = ''.join([str(i) for i in cumulative_result])

        if op == Operator.ADD or op == Operator.ADDCAR:
            if carry:
                self.add_carry.append(carry)
            system = 2
            self.s1_scratchpad = None
            if primary:
                self.intermediate_str = "".join([str(i) for i in intermediate])


        if op == Operator.MUL or op == Operator.MULCAR:
            self.mul_carry.append(carry)
            if primary:
                self.intermediate_str = "|".join(["".join(map(str, i)) for i in intermediate])


        add_carry = "".join(str(i) for i in self.add_carry)
        
        mul_carry = "".join(str(i) for i in self.mul_carry)


        self.s2_carry = f"{op_to_symbol(op)}|{mul_carry}|{add_carry}"

    
        self.s1_scratchpad = (
            self.problem
            + "|"
            + self.intermediate_str
            + "="
            + cumulative_result
        )
        self.s1_result_only = (
            self.problem + "=" + cumulative_result
        )
        self.s2_carry_scratch_pad = self.s2_carry + "|" + self.intermediate_str

        


        result = Result(
            operator=op_to_symbol(op),
            step=self.step_counter,
            system=system,
            cur_prob=cur_prob,
            s1_result=self.s1_result_only,
            s1_scratchpad=self.s1_scratchpad,
            s2_carry=self.s2_carry,
            s2_carry_scratchpad=self.s2_carry_scratch_pad,
            cumulative_result=cumulative_result,
            final_result=self.final_result,
            problem=self.problem,
        )


        self.step_counter += 1
        self.results.append(result)

    def get_final_result(self):
        return self.results


def int_to_list(integer, reverse=False) -> list:
    int_list = [int(i) for i in [*str(integer)]]
    return int_list[::-1] if reverse else int_list


def list_to_int(_list, reverse=False) -> int:
    if reverse:
        _list.reverse()
    return int("".join(map(str, _list[::-1])))


def left_shift(rows):
    # Extend zeros for each row to align left shift

    rows = rows.copy()
    for idx, row in enumerate(rows):
        rows[idx] = [0] * idx + row

    return rows


def add(addition_rows, writer=None, log=False, cumulative_result=[], primary=False):
    problem = "+".join([str(list_to_int(i)) for i in addition_rows])

    if not writer:
        writer = StepWriter(problem=problem, op=Operator.ADD)



    result = []
    carry = 0

    maximum_col = max(len(i) for i in addition_rows)

    for index in range(maximum_col):
        column = [lst[index] for lst in addition_rows if index < len(lst)]
        column_sum = sum(column) + carry
        result.append(column_sum % 10)
        carry = column_sum // 10

        if cumulative_result and index >= len(cumulative_result) and primary:
            # Start appending to final result if prior multiplications have completed.
            cumulative_result.append(result[-1])

        if log:
            writer.write_step(
                op=Operator.ADD,
                cur_prob=problem,
                intermediate=result,
                carry=carry,
                cumulative_result=cumulative_result,
                primary=primary, # Indicate whether add is the main operation taking place
            )

    while carry > 0:
        result.append(carry % 10)
        cumulative_result.append(carry % 10)
        # print(cumulative_result)
        if log:
            writer.write_step(
                op=Operator.ADDCAR,
                cur_prob=problem,
                intermediate=result,
                carry=carry,
                cumulative_result=cumulative_result,
                primary=primary # Indicate whether add is the main operation taking place
            )
        carry //= 10

    return list_to_int(result), cumulative_result


def multiply(multiplier, multiplicand, log=False, writer=None, final_result=[]):

    problem = f"{multiplier}*{multiplicand}"

    m1_digits = int_to_list(multiplier, reverse=True)
    m2_digits = int_to_list(multiplicand, reverse=True)

    result_rows = []
    carry = 0
    cumulative_result = []

    for i, m1 in enumerate(m1_digits):
        result_rows.append([])
        carry = 0
        for j, m2 in enumerate(m2_digits):

            out = m1 * m2
            if carry:
                c = int_to_list(carry)
                o = int_to_list(out, reverse=True)
                out, _ = add([c, o], log=True, writer=writer, cumulative_result=cumulative_result)
            carry = out // 10
            result = out % 10
            result_rows[i].append(result)

            

            if log:
                writer.write_step(
                    op=Operator.MUL,
                    cur_prob=problem,
                    intermediate=result_rows,
                    carry=carry,
                    cumulative_result=cumulative_result,
                )

            if j == 0:
                shift = left_shift(result_rows)
                added, _ = add(shift, log=True, writer=writer, cumulative_result=cumulative_result)
                computed = int_to_list(added, reverse=True)

                try:
                    cumulative_result.append(computed[len(cumulative_result)])
                except IndexError as e:
                    # Prior result contained all zeros, python evaluates 0000 -> 0.
                    cumulative_result.append(0)

        if carry:
            result_rows[i].append(carry)

            if log:
                writer.write_step(
                    op=Operator.MULCAR,
                    cur_prob=problem,
                    carry=carry,
                    intermediate=result_rows,
                    cumulative_result=cumulative_result,
                )

    return result_rows, final_result


def full_multiply(multiplier, multiplicand):

    final_result = multiplier * multiplicand

    writer = StepWriter(
        problem=f"{multiplier}*{multiplicand}",
        op=Operator.MUL,
        final_result=final_result,
    )

    result_rows, final_result = multiply(
        multiplier, multiplicand, writer=writer, log=True
    )
    shifted_rows = left_shift(result_rows)
    added_row = add(shifted_rows, cumulative_result=final_result, log=False, writer=writer, primary=True)
    return added_row, writer.get_final_result()


def test(floor, ceiling, num_tests=10):

    for _ in range(num_tests):
        fin_res = []
        m1 = random.randint(floor, ceiling)
        m2 = random.randint(floor, ceiling)
        addition_rows, fin_res = multiply(m1, m2, log=False)
        shifted_rows = left_shift(addition_rows)
        added_row, fin_res = add(shifted_rows, final_result=fin_res, log=False)
        true_added = true_add(shifted_rows)

        cumulative_result = list_to_int(fin_res)
        assert added_row == true_added
        assert true_added == m1 * m2

        assert cumulative_result == true_added


def true_add(addition_matrix):
    numbers = [list_to_int(i) for i in addition_matrix]
    return sum(numbers)




def make_data(csv_out, num_examples, min_num, max_num):
    result_fields = fields(Result)
    field_names = ["example"] + [i.name for i in result_fields] 

    with open(csv_out, mode='w', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writeheader()

        for i in tqdm.tqdm(range(num_examples), desc=f"Writing {csv_out}"):

            m1 = random.randint(min_num, max_num)
            m2 = random.randint(min_num, max_num)

            _, steps = full_multiply(m1, m2)

            for item in steps:
                example = {"example": i}
                example.update(asdict(item))

                csv_writer.writerow(example)




def merge_csvs(in_files, out_file):
    """
    Merge the CSV files in in_files into a single CSV file at out_file.

    The CSV files must have the same header, which we read from the first file.
    """
    
    with open(in_files[0], 'r') as f_in:
        csv_reader = csv.DictReader(f_in)
        headers = csv_reader.fieldnames

    with open(out_file, 'w') as f_out:
        csv_writer = csv.DictWriter(f_out, fieldnames=headers)
        csv_writer.writeheader()
        for in_file in in_files:
            with open(in_file, 'r') as f_in:
                csv_reader = csv.DictReader(f_in)
                for row in csv_reader:
                    csv_writer.writerow(row)


data_dir = 'prototyping/07-bptt-math/data'



lower_bound_train = os.path.join(data_dir, 'lower_bound_train.csv')
make_data(lower_bound_train, 40000, 0, 6000)

upper_bound_train = os.path.join(data_dir, 'upper_bound_train.csv')
make_data(upper_bound_train, 10000, 8000, 10000)

train_dataset = os.path.join(data_dir, 'train_dataset.csv')
merge_csvs([lower_bound_train, upper_bound_train], train_dataset)

interpolation_test = os.path.join(data_dir, 'interpolation_test.csv')
make_data(interpolation_test, 5000, 6000, 8000)

extrapolation_test = os.path.join(data_dir, 'extrapolation_test.csv')
make_data(extrapolation_test, 5000, 10000, 12000)

 



