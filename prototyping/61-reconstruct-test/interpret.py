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
        | "(" "I" expr expr expr ")" -> iff
        | "(" "F" defn "0" expr ")" -> forr
    ?defn: "[" var expr "]"
    ?op: "+"   -> add
       | "*"   -> mul
       | "<"   -> lt
    ?obj: var | digit
    var: /[a-cx-z]/
    digit: /[0-9]/
"""

#        | "="   -> eq
#        | ">"   -> gt 


@lark.visitors.v_args(meta=True)
class AddMultLetInterpreter(lark.visitors.Interpreter):
    "Top-down parser for the grammar"

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = [{}]
        self.intermediates = []
        self.positions = []

    def _getvar(self, var):
        for state in reversed(self.state):
            if var in state:
                return state[var]
        raise Exception(f"Unknown variable {var}")

    def calc(self, meta, children):
        op, e1, e2 = children
        op = self.visit(op)
        e1,e1s = self.visit(e1)
        e2,e2s = self.visit(e2)
        
        # intentional duplication: in case future ops have different shapes
        if op == '+':
            res = (e1 + e2) % 10
        elif op == '*':
            res = (e1 * e2) % 10
        elif op == '=':
            res = int(e1 == e2)
        elif op == '>':
            res = int(e1 > e2)
        elif op == '<':
            res = int(e1 < e2)
        else:
            raise Exception("Unknown operator")

        sublabels = '__'+e1s+e2s+str(res)
        self.intermediates.append(f'({op}{e1}{e2})={res}')
        self.positions.append( (meta.start_pos, meta.end_pos, res, op) )
        return (res, sublabels)
        
    def let(self, meta, children):
        defn, expr = children

        self.state.append({}) # new closure
        _, _, s_defn = self.visit(defn)
        result, s_expr = self.visit(expr)
        self.state.pop() # exit closure
        self.positions.append( (meta.start_pos, meta.end_pos, result, 'L') )
        return (result, '__'+s_defn+s_expr+str(result))
    
    def defn(self, meta, children):
        var, expr = children
        var = var.children[0].value
        val, s_exp = self.visit(expr)

        self.state[-1][var] = val
        #self.positions.append( (meta.start_pos, meta.end_pos, f"{var}{val}", 'D') )
        return (var, val, '__'+s_exp+str(val))

    def iff(self, meta, children):
        cond, e1, e2 = children
        cond, s_cond = self.visit(cond)
        e1, s_e1 = self.visit(e1)
        e2, s_e2 = self.visit(e2)

        val = e1 if cond else e2
        self.positions.append( (meta.start_pos, meta.end_pos, val, 'I') )
        return (val, '__'+s_cond+s_e1+s_e2+str(val))

    def forr(self, meta, children):
        """
        this is not a normal for loop: it has an accumulator variable,
        which each loop iteration adds to, and the final value is returned.
        rationale: useful test case for recursion.
        functional languages don't have mutable variables,
        so this is the next best thing.
        """
        defn, expr = children

        self.state.append({}) # new closure
        var_for, val_for, s_defn = self.visit(defn)

        # count from val_for down to zero.
        # it always runs at least once
        self.state[-1][var_for] = val_for
        accum = 0

        # mark start of for loop
        self.positions.append( (meta.start_pos, meta.end_pos, 0, 'E') )
        for i in range(val_for,-1,-1):
            one_loop_result, s_expr = self.visit(expr)
            self.state[-1][var_for] -= 1
            accum = (accum + one_loop_result) % 10
            #self.intermediates.append(f'(+{accumulator}{one_loop_result})={res}')
            #accumulator = res

            if i > 0:
                # non-final loops: Mark with "f" to distinguish from final loop
                # used to flag to reset loop state
                self.positions.append( (meta.start_pos, meta.end_pos, (i,accum), 'f') )
            else:
                # final loops: Mark with "F"
                # we can now remove the for node from the final expression
                self.positions.append( (meta.start_pos, meta.end_pos, (i,accum), 'F') )


        self.state.pop() # exit closure
        return (accum, '__'+s_defn+'_'+s_expr+str(accum))

    def var(self, meta, children):
        var = children[0].value
        val = self._getvar(var)
        self.positions.append( (meta.start_pos, meta.end_pos, val, 'V') )
        return (self._getvar(var), '_')

    def add(self, meta, children):
        return '+'
    def mul(self, meta, children):
        return '*'
    def eq(self, meta, children):
        return '='
    def gt(self, meta, children):
        return '>'
    def lt(self, meta, children):
        return '<'

    def digit(self, meta, children):
        n = int(children[0].value)
        return (n, '_')



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

    parser = lark.Lark(calc_grammar, start='expr', propagate_positions=True)
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

    # play code
    parser = lark.Lark(calc_grammar, start='expr', propagate_positions=True)
    interpreter = AddMultLetInterpreter()

    testexp = "(<(F[a(+(I639)(*10))]0(<(L[ca](+ca))(+(+aa)(*aa))))(F[a(L[b9](*bb))]0(*(I(*0a)(*aa)(+8a))(L[ya](+y6)))))"
    # testexp = "(L [x4] (L [y(+1(*2x))] (*y3)) )" # x=4, y=2x+1, 3y

    # testexp = testexp.replace(" ", "") # whitespace is for humans
    testtree = parser.parse(testexp)
    testans = interpreter.visit(testtree)

    seq = interpreter.positions

    print(testexp)
    cur_exp = testexp
    for x,y,z,op in seq:    
        a = list(cur_exp)
        a[x:y] = [' '] * (y-x)
        a[y-len(str(z)):y] = str(z)
        cur_exp = ''.join(a)
        print(f"{cur_exp} {op}")



    # n_items = 100
    # n_processes = 1
    # processes = []
    # file_prefix = 'temp-'

    # if n_processes == 1: # for ease of debugging
    #     make_data(f'{file_prefix}00.csv', n_items, 1, 10, 2**16)
    # else:
    #     items_per_proc = n_items // n_processes
    #     for i in range(n_processes):
    #         p = Process(target=make_data, args=(f'{file_prefix}{i:02}.csv', items_per_proc, 2, 10, 2**16))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    #     merge_csvs([f'{file_prefix}{i:02}.csv' for i in range(n_processes)], f'{file_prefix}all.csv')



