import generate, interpret
import lark
from hypothesis.extra.lark import from_lark
import csv, re
from multiprocessing import Process
import random

def make_data(out_file, n_examples, max_height, min_height=1, stepwise=False, enable_for=True, enable_let=True, enable_if=True, step_vars=False):
    """
    Generate problems based on the grammar, and write them to a CSV file.
    - out_file: path to the output CSV file
    - n_examples: how many examples to generate in total
    - max_height: maximum height of the generated examples
    Generates a roughly balanced number of examples for each height.
    """
    # setup
    parser = lark.Lark(interpret.calc_grammar, start='expr', propagate_positions=True)
    interpreter = interpret.AddMultLetInterpreter()

    fieldnames = ['height', 'width', 'example', 'answer', 'ans_mod10', 'ans_sublabels', 'tree_sig']
    if stepwise:
        fieldnames = ['idx', 'step', 'is_done'] + fieldnames + \
            ['cur_state', 'next_state', 
             'cur_action_aligned', 'cur_action_type', 'cur_action_tight', 'cur_action_res',
             'next_action_aligned', 'next_action_type', 'next_action_tight', 'next_action_res']

    gen_weights = generate.gen_weights
    if not enable_for:
        gen_weights[4] = 0
    if not enable_let:
        gen_weights[3] = 0
    if not enable_if:
        gen_weights[2] = 0

    # skip these states when generating stepwise data
    skip_states = ['E'] # used internally to manage loop stack
    if not step_vars:
        skip_states.append('V') # variable lookup steps: (+ab) -> (+2b) -> (+21)

    with open(out_file, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        idx = 0
        n_items_per_height = n_examples // (max_height - min_height + 1)
        # iterate backwards: from max_height to min_height
        # reason: pandas/hf CSV parsers infer dtypes from first few rows, 
        # they get confused by the height=1 examples (all ints)
        for j,height in enumerate(range(max_height, min_height-1, -1)):
            for i in range(n_items_per_height):
                interpreter.reset()
                data = {}
                data['example'] = generate.generate_program(height-1, list(), gen_weights=gen_weights) # h-1 to align with other datasets and interpret.get_height
                tree = parser.parse(data['example'])
                data['answer'], data['ans_sublabels'] = interpreter.visit(tree)
                data['height'] = interpret.get_height(tree)
                data['width']  = len(data['example'])

                data['ans_mod10'] = data['answer'] % 10
                data['tree_sig'] = re.sub(r'\d', 'd', data['example'])
                data['tree_sig'] = re.sub(r'[\+\*\<]', 'o', data['tree_sig'])
                data['tree_sig'] = re.sub(r'[a-cx-z]', 'v', data['tree_sig'])
                
                if stepwise:
                    data['idx'] = idx
                    sequence = interpreter.positions

                    # first, emit a virtual initial state
                    # trains the model to fill next_state from raw example
                    step_prev = { # we CSV by one step so we can add next_action later
                        **data, 
                        **{'step': 0, 'is_done': int(len(sequence) == 0), # no steps if no changes
                           'cur_state': ' '*len(data['example']), 'next_state': data['example'],
                           'cur_action_aligned': ' '*len(data['example']), 'cur_action_res': ' ',
                           'cur_action_tight': ' ', 'cur_action_type': ' ',}
                    }

                    cur_state = data['example'] # entire legible problem state: only updated with complete formal state changes
                    cur_working_state = data['example'] # dirty working copy of cur_state (e.g. some steps produce partial formal state changes, tracked here)
                    k = 1
                    loop_stack = []

                    for c,(x,y,res,op) in enumerate(sequence):
                        if op == 'D': 
                            continue # skip obvious steps

                        step_data = {
                            'step': k,
                            'is_done': int(c == len(sequence)-1),
                            'cur_state': cur_state, # updated with cur_exp at each loop
                            'cur_action_aligned': ' '*len(data['example']), # default to blank
                            'cur_action_res': ' ', # default to blank
                            'cur_action_tight': ' ', # default to blank
                            'cur_action_type': ' ', # default to blank
                            'next_action_aligned': ' '*len(data['example']), # default to blank
                            'next_action_res': ' ', # default to blank
                            'next_action_tight': ' ', # default to blank
                            'next_action_type': ' ', # default to blank
                        }

                        # current action is based on current working state
                        if op in ['+', '*', '<', '=', '>', 'I', 'L']:
                            step_data['cur_action_tight'] = cur_working_state[x:y].replace(' ', '')
                            step_data['cur_action_aligned'] = ' '*x + cur_working_state[x:y] + ' '*(len(cur_working_state)-y)
                            step_data['cur_action_type'] = op
                            step_data['cur_action_res'] = str(res)

                        cws_list = list(cur_working_state) # listify for easy updates. 
                        # (invariant: no changes to non-list version within this if)
                        if op == 'E': # start of a for loop
                            # push loop inner expr to stack
                            # so later steps can reset
                            lexpr_start = cur_state[x:y].find(']') + 2 + x # from after accumulator
                            lexpr_end = y - 1
                            lexpr = cur_state[lexpr_start:lexpr_end]
                            loop_stack.append((lexpr, lexpr_start, lexpr_end))
                        elif op == 'f': # end of non-final loop iteration
                            #print(f"For loop: {x}, {y}, {res}, {op}")
                            cur_for_i, cur_for_acc = res
                            for_update_spot = cur_working_state[x:y].find(']') + x
                            cws_list[for_update_spot-1] = str(cur_for_i-1)
                            cws_list[for_update_spot+1] = str(cur_for_acc)
                            # reset loop inner expr
                            lexpr, lexpr_start, lexpr_end = loop_stack[-1]
                            cws_list[lexpr_start:lexpr_end] = lexpr
                        elif op == 'F': # finished for loop
                            cws_list[x:y] = [' '] * (y-x)
                            cws_list[y-len(str(res[1])):y] = str(res[1])
                            loop_stack.pop() # pop loop stack
                        else: # general case
                            cws_list[x:y] = [' '] * (y-x)
                            cws_list[y-len(str(res)):y] = str(res)

                        # ends of for loops emit a virtual accumulator add
                        if op in ['f', 'F']:
                            # remove the inner [defn] since it's used up
                            # system 1 should see this as an add
                            let_start = cur_state[x:y].find('[') + x
                            let_end = cur_state[x:y].find(']') + 1 + x
                            action_nodefn = ' '*x + cur_working_state[x:let_start] + ' '*(let_end-let_start) + cur_working_state[let_end:y] + ' '*(len(cur_working_state)-y)
                            action_nodefn = action_nodefn.replace("F", "+")
                            
                            step_data['cur_action_tight'] = action_nodefn.replace(' ','') 
                            step_data['cur_action_aligned'] = action_nodefn
                            step_data['cur_action_type'] = '+'
                            step_data['cur_action_res'] = str(res[1])

                        cur_working_state = ''.join(cws_list)
                        #print(f"Next state: {cur_working_state}")
                        step_data['next_state'] = cur_working_state

                        if op in skip_states: 
                            # updates which don't emit a formal state change, or action
                            # V steps: trivial action, change state but bundle in other states
                            # E steps: only used in this function, to manage loop stack
                            continue
                        else: # formal step
                            cur_state = cur_working_state # update formal state

                            # write next_action data to previous step
                            if 'cur_action_type' in step_data: # _aligned is always present
                                step_prev['next_action_aligned'] = step_data['cur_action_aligned']
                                step_prev['next_action_type'] = step_data['cur_action_type']
                                step_prev['next_action_tight'] = step_data['cur_action_tight']
                                step_prev['next_action_res'] = step_data['cur_action_res']

                            # sanity checks
                            assert(len(step_prev['example']) == len(step_prev['cur_state']))
                            assert(len(step_prev['example']) == len(step_prev['next_state']))
                            assert(len(step_prev['example']) == len(step_prev['cur_action_aligned']))
                            # not all steps have a next action (e.g. state-only changes)
                            # assert(len(step_prev['example']) == len(step_prev['next_action_aligned']))


                            # write a step to CSV (buffered by 1 step)
                            csv_writer.writerow(step_prev)
                            step_prev = {**data, **step_data}
                            k += 1
                    csv_writer.writerow(step_prev) # emit final step
                else:
                    csv_writer.writerow(data)
                assert(len(data['example']) == len(data['ans_sublabels']))
                if i % 500 == 0:
                    print(f'wrote {i}/{n_items_per_height} examples for height {height} ({j}/{max_height-min_height+1} heights)')
                idx += 1


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



def make_data_mp(out_file, n_examples, max_height, min_height=1, n_processes=1, stepwise=False, enable_for=True, enable_let=True, enable_if=True, step_vars=True):
    "MP wrapper"
    processes = []
    filenames = []

    if n_processes == 1: # passthrough for nicer debugging
        make_data(out_file, n_examples, max_height, min_height, stepwise, enable_for, enable_let, step_vars)        
    else:
        items_per_proc = n_examples // n_processes
        for i in range(n_processes):
            filenames.append(f'{out_file}-{i:02}.csv')

            p = Process(target=make_data, args=(filenames[-1], items_per_proc, max_height, min_height, stepwise, enable_for, enable_let, enable_if, step_vars))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        merge_csvs(filenames, out_file)







if __name__ == '__main__':
    random.seed(42)
    make_data_mp('../data/test-step-amli-500k-d4.csv', 500000, min_height=2, max_height=4, n_processes=8, stepwise=True, enable_for=False, enable_let=True, enable_if=True, step_vars=True)

    # # setup
    # parser = lark.Lark(interpret.calc_grammar, start='expr')
    # interpreter = interpret.AddMultLetInterpreter()

    # # Generate toy programs
    # max_depth = 1
    # n_programs = 1000000
    # toy_programs = [generate.generate_program(max_depth, list()) for _ in range(n_programs)]

    # for program in toy_programs:
    #     interpreter.reset()
    #     print(f"Program:   {program}")
    #     tree = parser.parse(program)
    #     res, sublabels = interpreter.visit(tree)

    #     print(f"Sublabels: {sublabels}")
    #     print(f"Lark result: {res}")
    #     print(f"Depth: {interpret.get_height(tree)}")
    #     print(f"Intermediates: {interpreter.intermediates}\n")



