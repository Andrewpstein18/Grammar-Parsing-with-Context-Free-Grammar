"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Andrew Stein
UNI: APS2231
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """

        # Step 1: Create a dictionary
        parsing_table = defaultdict(list)

        # Step 2: Fill in the basic values of the sentence
        i = 0
        for token in tokens:
            # list of tuples
            set_of_nt = self.grammar.rhs_to_rules[(str(token),)]
            for tuples in set_of_nt:
                nt, token, prob = tuples
                parsing_table[(i, i + 1)].append(nt)
            i += 1

        # Step 3: Fill in the more complicated parts of the table
        for length in range(2, len(tokens) + 1):
            for i in range(0, len(tokens) - length + 1):
                j = length + i
                for k in range(i + 1, j):
                    np_from_i = parsing_table[(i, k)]
                    np_from_j = parsing_table[(k, j)]
                    for np_i in np_from_i:
                        for np_j in np_from_j:
                            potential_tuple = (np_i, np_j)
                            rule_list = self.grammar.rhs_to_rules[potential_tuple]
                            for rule in rule_list:
                                nt, product, prob = rule
                                parsing_table[(i, j)].append(nt)

        for symbol in parsing_table[(0, len(tokens))]:
            if symbol == grammar.startsymbol:
                return True

        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """

        # Step 1: Create the dictionaries
        parsing_table = defaultdict(list)
        probability_table = defaultdict(list)

        # Step 2: Fill in the basic values of the sentence for both tables
        i = 0
        for token in tokens:
            # list of nt that map to terminals
            set_of_nt = self.grammar.rhs_to_rules[(str(token),)]

            # create the dictionaries for each location
            backpointer_dictionary_for_location = defaultdict(list)
            probability_dictionary_for_location = defaultdict(list)

            # for each of the non-terminals
            for tuples in set_of_nt:
                nt, token, prob = tuples

                # add each of the non-terminals to the location spot of each location in the origin
                backpointer_dictionary_for_location[nt] = token[0]
                probability_dictionary_for_location[nt] = math.log(prob)

                # place those dictionaries inside each of the locations
                parsing_table[(i, i + 1)] = backpointer_dictionary_for_location
                probability_table[(i, i + 1)] = probability_dictionary_for_location

            # increment to the next location
            i += 1

        # Step 3: Fill in the non-terminals based on other non-terminals
        # Start with looping through the length
        for length in range(2, len(tokens) + 1):
            # position the i
            for i in range(0, len(tokens) - length + 1):
                # create the dictionaries that hold the ways to get to each non-terminal
                bp_dict_for_location = defaultdict(list)
                prob_dict_for_location = defaultdict(list)

                # declare j, loops through with i
                j = length + i

                # loop through k
                for k in range(i + 1, j):
                    # Extract the potential nts from i
                    for nt_i in parsing_table[(i, k)]:
                        # Extract the potential nts from j
                        for nt_j in parsing_table[(k, j)]:
                            # for each cartesian product pair check if there is a nt that maps to it
                            potential_tuple = (nt_i, nt_j)
                            # extract all nts
                            rule_list = self.grammar.rhs_to_rules[potential_tuple]
                            # for each nt, add to the parsing table
                            for rule in rule_list:
                                nt, x, prob = rule
                                lhs = (nt_i, i, k)
                                rhs = (nt_j, k, j)

                                # keep the path with the highest probability
                                if nt not in prob_dict_for_location or math.log(prob) > prob_dict_for_location[nt]:
                                    # update the source for both charts if not in or higher probability
                                    prob_dict_for_location[nt] = math.log(prob)
                                    bp_dict_for_location[nt] = (lhs, rhs)

                        # if there is no path, place in an empty dictionary
                        parsing_table[(i, j)] = (bp_dict_for_location)
                        probability_table[(i, j)] = (prob_dict_for_location)

                # add the dictionaries to the location of the tables
                if (i, j) not in parsing_table:
                    parsing_table[(i, j)] = defaultdict(list)
                    probability_table[(i, j)] = defaultdict(list)

        # return both tables
        return parsing_table, probability_table

def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """

    # check that the location exists and that the type is a tuple (which means that this is a nt)
    if type(chart[(i, j)][nt]) == tuple:
        # extract the path of the creation of that nt
        i_source, j_source = chart[(i, j)][nt]
        i_nt, i_i, i_j = i_source
        j_nt, j_i, j_j = j_source

        # recursively build the string from the sources
        recursive_string = nt, get_tree(chart, i_i, i_j, i_nt), get_tree(chart, j_i, j_j, j_nt)
    # this means that the value is a terminal
    else:
        terminal = chart[(i, j)][nt]
        # if the terminal does not exist, throw an error
        if len(terminal) == 0:
            raise KeyError('Key is not in grammar')
        recursive_string = nt, terminal

    # return the recursively built string
    return recursive_string


if __name__ == "__main__":
    with open('./atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['with', 'the', 'least', 'expensive', 'fare', '.']
        print(parser.is_in_language(toks))
        table, prob = parser.parse_with_backpointers(toks)
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        assert check_table_format(table)
        assert check_probs_format(prob)
