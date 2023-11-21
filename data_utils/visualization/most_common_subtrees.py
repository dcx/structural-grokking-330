from collections import defaultdict
import pandas as pd

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

def parse_expression(expression):
    """
    Parse the mathematical expression into a tree structure.
    """
    stack = []
    current_node = None
    number_buffer = []  # Buffer to handle multi-digit numbers

    for char in expression:
        if char.isdigit():  # If it's a number, we add it to the number buffer
            number_buffer.append(char)
        elif char == '(':
            if number_buffer:
                # Flush the number buffer to create a new node
                node = Node(int(''.join(number_buffer)))
                number_buffer = []
                if current_node:
                    current_node.add_child(node)
                else:
                    # This should only happen for the very first number
                    current_node = node
            if current_node:
                stack.append(current_node)
            current_node = None
        elif char in '+*':
            if number_buffer:
                # Flush the number buffer to create a new node
                node = Node(int(''.join(number_buffer)))
                number_buffer = []
                if current_node:
                    current_node.add_child(node)
            current_node = Node(char)
        elif char == ')':
            if number_buffer:
                # Flush the number buffer to create a new node
                node = Node(int(''.join(number_buffer)))
                number_buffer = []
                current_node.add_child(node)
            if stack:
                node = stack.pop()
                node.add_child(current_node)
                current_node = node
    return current_node

def serialize_tree(node):
    """
    Serialize the tree into a string representation.
    """
    if node is None:
        return '#'
    return f'{node.value}({serialize_tree(node.children[0]) if node.children else "#"})' \
           f'({serialize_tree(node.children[1]) if len(node.children) > 1 else "#"})'

def find_subtrees(node, subtree_count):
    """
    Find all subtrees and count their occurrences.
    """
    if node is None:
        return '#'
    left = find_subtrees(node.children[0], subtree_count) if node.children else '#'
    right = find_subtrees(node.children[1], subtree_count) if len(node.children) > 1 else '#'
    subtree = f'{node.value}({left})({right})'
    subtree_count[subtree] += 1
    return subtree

# Load the dataset
data_path = '/home/x11kjm/structural-grokking/data_utils/ds_addmult_mod10_data/data-addmult-231019.csv'
data = pd.read_csv(data_path)

# Process each expression in the dataset
subtree_count = defaultdict(int)
for expression in data['example']:
    tree = parse_expression(expression)
    find_subtrees(tree, subtree_count)

# Sort the subtrees by their frequency
sorted_subtrees = sorted(subtree_count.items(), key=lambda item: item[1], reverse=True)

# Print the most common subtrees
for subtree, count in sorted_subtrees[:5]:  # Let's print the top 5 for brevity
    print(f'Subtree: {subtree}, Count: {count}')