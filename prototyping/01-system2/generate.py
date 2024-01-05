# Define a function to generate simple toy programs based on the given grammar

import random, copy
from datetime import datetime

var_names = ['x', 'y', 'z', 'a', 'b', 'c']

def generate_program(tgt_depth, closed_vars, min_depth=None):
    """
    tgt_depth: the target depth of the generated program
        the final program will never be deeper than this
        but may be shallower if the random choices lead to it
    closed_vars: a list of dictionaries, each representing a closure
        just give an empty list to start
    min_depth: if set, all subtrees will be at least this deep.
        Prevents the generation of trivial programs like "2"
    """
    if min_depth is None:
        min_depth = tgt_depth

    # Randomly choose an expression type
    choices, weights = [], []
    if tgt_depth >= 0 and min_depth<=0: # base case
        choices += ['number']
        weights += [1]
    if tgt_depth >= 1:
        choices += ['operation', 'if']
        weights += [30, 6] # 30, 15]
    if tgt_depth >= 2: # these have a builtin nested expression
        choices += ['let', 'for']
        weights += [30, 15]

    # Only allow variables if there are variables to choose from
    if len(closed_vars) > 0 and tgt_depth>=0 and min_depth<=0:
        choices.append('var')
        weights.append(3)

    expr_type = random.choices(choices, weights)[0]


    # Generate a number expression
    if expr_type == 'number':
        rand_num = random.randint(0, 9)
        return str(rand_num)

    # Generate a variable expression
    elif expr_type == 'var':
        # Choose a random variable from the list of closed variables
        closure = random.choice(closed_vars)
        var = random.choice(list(closure.keys()))
        val = closure[var]
        return var       

    # Generate an operation expression
    elif expr_type == 'operation':
        op = random.choice(['+', '*', '<']) # '=', '>', 
        e1s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)
        e2s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)

        return f"({op}{e1s}{e2s})" #, res)



    # Generate a let expression
    elif expr_type == 'let':
        var = random.choice(var_names)
        e1s = generate_program(tgt_depth-2, closed_vars.copy(), min_depth-2)

        # Add the variable to the list of closed variables
        closed_vars.append({var: None}) # new closure
        e2s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)
        closed_vars.pop() # exit closure

        return f"(L[{var}{e1s}]{e2s})" #, e2v)

    # Generate an if expression
    elif expr_type == 'if':
        e1s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)
        e2s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)
        e3s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1)
        return f"(I{e1s}{e2s}{e3s})"

    # # Generate a do expression
    # elif expr_type == 'do':

    #     l_ens = []
    #     n_expressions = random.randint(2, 5)
    #     for i in range(n_expressions):
    #         ens, env = generate_program(max_depth-1, closed_vars, min_depth-1)
    #         l_ens.append(ens)

    #     exprs = ' '.join(l_ens)
    #     return (f"(D{exprs})", env) # return the last expression's value

    # Generate a for expression
    elif expr_type == 'for':
        var = random.choice(var_names)

        e1s = generate_program(tgt_depth-2, closed_vars.copy(), min_depth-2)

        closed_vars.append({var: 0}) # new closure with loop var init
        e2s = generate_program(tgt_depth-1, closed_vars.copy(), min_depth-1) # loop body

        for_expr = f"(F[{var}{e1s}]0{e2s})"

        return for_expr



if __name__ == "__main__":
    # Generate toy programs
    max_depth = 3
    toy_programs = [generate_program(max_depth, list()) for _ in range(2)]
    for program in toy_programs:
        print(f"Program: {program}")





