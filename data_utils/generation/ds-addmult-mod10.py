import csv, re
from lark import Lark, Transformer
from hypothesis.extra.lark import from_lark
from multiprocessing import Process

calc_grammar = """
    ?expr: digit
        | "(" op expr expr ")" -> calc
    digit: /[0-9]/
    ?op: "+"   -> add
       | "*"   -> mul
"""
# Saved for later
#      | "t"   -> ten



class CalcInterpreter(Transformer):
    "Interpret the parse tree as a calculator"

    def digit(self, n):
        (n,) = n
        return int(n)

    def add(self, s):
        return '+'
    def mul(self,s):
        return '*'
    #def ten(self,s):
    #    return 't'

    def op(self, op):
        (op,) = op

    def calc(self, children):
        op, e1, e2 = children
        if op == '+':
            return e1 + e2
        elif op == '*':
            return e1 * e2
        #elif op == 't':
        #    return 10*e1 + e2
        
        raise Exception("Unknown operator")


def get_height(tree):
    "Get the height of a Lark tree (height 1 = single digit)"
    if hasattr(tree, 'children') and len(tree.children) > 0:
        return 1 + max([get_height(c) for c in tree.children])
    else:
        return 0

def run_calculator(expression):
    parser = Lark(calc_grammar, start='expr')
    interpreter = CalcInterpreter()
    
    try:
        tree = parser.parse(expression)
        result = interpreter.transform(tree)
        return result
    except Exception as e:
        return f"An error occurred: {e}"

def make_data(out_file, n_examples, min_height, max_height, max_ans):
    """
    Generate problems based on the grammar, and write them to a CSV file.
    - out_file: path to the output CSV file
    - n_examples: how many examples to generate in total
    - min_height: minimum height of the generated examples
    - max_height: maximum height of the generated examples
    - exclude_strings: if not None, separate out examples that contain one of these strings
    - exclude_file: Write the excluded examples to this file
    """

    parser = Lark(calc_grammar, start='expr')
    interpreter = CalcInterpreter()
    generator = from_lark(parser, start="expr")

    n_good = 0
    with open(out_file, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['height', 'width', 'example', 'answer', 'ans_mod10', 'tree_sig'])
        csv_writer.writeheader()

        while True:
            example = generator.example()
            tree = parser.parse(example)
            height = get_height(tree)
            if height < min_height or height > max_height:
                continue
            else:
                answer = interpreter.transform(parser.parse(example))
                if answer > max_ans:
                    continue

                n_good += 1
                ans_mod10 = answer % 10
                tree_sig = re.sub(r'\d', 'd', example)
                tree_sig = re.sub(r'\+', 'o', tree_sig)
                csv_writer.writerow({'height': height, 'width': len(example), 'example': example, 'answer': answer, 'ans_mod10': ans_mod10, 'tree_sig': tree_sig})

                if n_good % 500 == 0:
                    print(f'wrote {n_good}/{n_examples} examples')
                if n_good >= n_examples:
                    break

def run_calculator(expression: str) -> int:
    parser = Lark(calc_grammar, start='expr')
    interpreter = CalcInterpreter()
    
    try:
        tree = parser.parse(expression)
        result = interpreter.transform(tree)
        return result
    except Exception as e:
        return f"An error occurred: {e}"


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



if __name__ == '__main__':

    n_items = 20000
    n_processes = 4
    processes = []
    file_prefix = 'data-addmult-231102-2m-'

    items_per_proc = n_items // n_processes
    for i in range(n_processes):
        p = Process(target=make_data, args=(f'{file_prefix}{i:02}.csv', items_per_proc, 1, 10, 2**16))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_csvs([f'{file_prefix}{i:02}.csv' for i in range(n_processes)], f'{file_prefix}all.csv')

    #make_data('data-addmult-231020-150k.csv', 150000, 1, 10, 2**16)


