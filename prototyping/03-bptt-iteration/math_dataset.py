from dataclasses import dataclass
from enum import Enum, auto
import logging
import random

from rich.logging import RichHandler


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

class Op(Enum):
    MUL = auto()
    ADD = auto()

@dataclass
class Step:
    operator: Op
    digits_involved: list
    carry: int
    intermediate_result: int
    final_result: int


def int_to_list(integer) -> list:
    return [int(i) for i in [*str(integer)]]

def add(adder, addend):
    adder = int_to_list(adder)[::-1]
    addend = int_to_list(addend)[::-1]
    
    length = max(len(adder), len(addend))
    adder += [0] * (length - len(adder))
    addend += [0] * (length - len(addend))

    result = []
    carry = 0
    for a, b in zip(adder, addend):
        total = a + b + carry
        carry = total // 10
        result.append(total % 10)
        logger.info(f"Adding {a} + {b} + carry {carry} = {total}, New carry = {total // 10}, Result so far = {result[::-1]}")

    if carry != 0:
        result.append(carry)
        logger.info(f"Final carry = {carry}, Result = {result[::-1]}")

    result = int(''.join(map(str, result[::-1])))
    return result

def multiply(multiplier, multiplicand):
    multiplier = int_to_list(multiplier)[::-1]
    multiplicand = int_to_list(multiplicand)[::-1]
    result = 0

    logger.info(f"Multiplier: {multiplier[::-1]}, Multiplicand: {multiplicand[::-1]}")
    for i, mler in enumerate(multiplier):
        for j, mand in enumerate(multiplicand):
            tens_power = 10 ** (i + j)
            partial_product = mler * mand * tens_power
            logger.info(f"Partial product: {mler} * {mand} * 10^{i + j} = {partial_product}")

            if partial_product < 100 and result == 0:
                logger.info(f"No need to add, result is {result}: set result to {partial_product} ")
                result += partial_product
            else:
                logger.info(f"Adding partial product: {partial_product} to total {result}")
                result = add(result, partial_product)

    return result


def test(min_digit, max_digit, num_tests=100):
    logger.setLevel(logging.ERROR)

    for _ in range(num_tests):

        er = random.randint(10**min_digit, 10**max_digit)
        nd = random.randint(10**min_digit, 10**max_digit)

        real_mul = er*nd
        emp_mul = multiply(er, nd)

        assert real_mul == emp_mul

        real_add = er+nd
        emp_add = add(er, nd)
        assert real_add == emp_add




    
# test(3, 7)

multiply(4, 7)

add(3,4)