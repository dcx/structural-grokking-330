# generator script for addmult-let
# addmult-mod10 with let expressions

import csv, re
import lark
from hypothesis.extra.lark import from_lark
from multiprocessing import Process

calc_grammar = """
    ?expr: obj
        | "(" op expr expr ")" -> calc
        | "(" "L" defn expr ")" -> let
    ?defn: "[" var expr "]"
    ?op: "+"   -> add
       | "*"   -> mul
    ?obj: var | digit
    var: /[x-z]/
    digit: /[0-9]/
"""

class AddMultLetInterpreter(lark.visitors.Interpreter):
    "Top-down parser for the grammar"

    def __init__(self):
        self.state = [{}]

    def reset(self):
        self.state = [{}]

    def _getvar(self, var):
        for state in reversed(self.state):
            if var in state:
                return state[var]
        raise Exception(f"Unknown variable {var}")

    def calc(self, tree):
        op, e1, e2 = tree.children
        op = self.visit(op)
        e1,e1s = self.visit(e1)
        e2,e2s = self.visit(e2)
        
        # intentional duplication: in case future ops have different shapes
        if op == '+':
            res = (e1 + e2) % 10
            return (res, '__'+e1s+e2s+str(res))
        elif op == '*':
            res = (e1 * e2) % 10
            return (res, '__'+e1s+e2s+str(res))
        raise Exception("Unknown operator")

    def let(self, tree):
        defn, expr = tree.children

        self.state.append({}) # new closure
        _, s_defn = self.visit(defn)
        result, s_expr = self.visit(expr)
        self.state.pop() # exit closure
        return (result, '__'+s_defn+s_expr+str(result))
    
    def defn(self, tree):
        var, expr = tree.children
        var = var.children[0].value
        val, s_exp = self.visit(expr)

        self.state[-1][var] = val
        return (None, '__'+s_exp+str(val))

    def var(self, tree):
        var = tree.children[0].value
        return (self._getvar(var), '_')

    def add(self, tree):
        return '+'
    def mul(self,tree):
        return '*'
    def digit(self, tree):
        n = int(tree.children[0].value)
        return (n, '_')

parser = lark.Lark(calc_grammar, start='expr')
interpreter = AddMultLetInterpreter()

testexp = "(L[x4](+x2))" # x=4, x+2
testexp = "(L [x4] (L [y(+1(*2x))] (*y3)) )" # x=4, y=2x+1, 3y

testexp = testexp.replace(" ", "") # whitespace is for humans
testtree = parser.parse(testexp)
testans = interpreter.visit(testtree)
print(testans)


def get_height(tree):
    "Get the height of a Lark tree (height 1 = single digit)"
    if hasattr(tree, 'children') and len(tree.children) > 0:
        return 1 + max([get_height(c) for c in tree.children])
    else:
        return 0

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

    parser = lark.Lark(calc_grammar, start='expr')
    interpreter = AddMultLetInterpreter()
    generator = from_lark(parser, start="expr")

    n_good = 0
    with open(out_file, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['height', 'width', 'example', 'answer', 'ans_mod10', 'ans_sublabels', 'tree_sig'])
        csv_writer.writeheader()

        while True:
            interpreter.reset()
            example = generator.example()
            tree = parser.parse(example)
            height = get_height(tree)
            if height < min_height or height > max_height:
                continue
            else:
                try:
                    answer, ans_inter = interpreter.visit(parser.parse(example))
                except:
                    continue
                if answer > max_ans:
                    continue

                n_good += 1
                ans_mod10 = answer % 10
                tree_sig = re.sub(r'\d', 'd', example)
                tree_sig = re.sub(r'\+', 'o', tree_sig)
                tree_sig = re.sub(r'[x-z]', 'v', tree_sig)
                csv_writer.writerow({'height': height, 'width': len(example), 'example': example, 'answer': answer, 'ans_mod10': ans_mod10, 'ans_sublabels': ans_inter, 'tree_sig': tree_sig})

                assert(len(example) == len(ans_inter))

                if n_good % 500 == 0:
                    print(f'wrote {n_good}/{n_examples} examples')
                if n_good >= n_examples:
                    break


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

    n_items = 1500000
    n_processes = 8
    processes = []
    file_prefix = 'data-addmultlet-'

    if n_processes == 1: # for ease of debugging
        make_data(f'{file_prefix}00.csv', n_items, 1, 10, 2**16)
    else:
        items_per_proc = n_items // n_processes
        for i in range(n_processes):
            p = Process(target=make_data, args=(f'{file_prefix}{i:02}.csv', items_per_proc, 2, 10, 2**16))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        merge_csvs([f'{file_prefix}{i:02}.csv' for i in range(n_processes)], f'{file_prefix}all.csv')



