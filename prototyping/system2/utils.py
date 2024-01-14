

def get_difference(seq1, seq2):

    difference = []
    for char1, char2 in zip(seq1, seq2):
        if char1 == char2:
            difference.append(' ')  # Append a space character if they match
        else:
            difference.append(char2)

    concat_difference = ''.join(difference)

    return concat_difference