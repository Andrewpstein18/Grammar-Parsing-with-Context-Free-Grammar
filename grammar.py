"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Andrew Stein
UNI: APS2231
"""
import math
import sys
from collections import defaultdict
from math import fsum


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """

        non_terminal_tracker = []

        # Step 1: Gather the non-terminals
        for nonTerminal in self.lhs_to_rules:
            non_terminal_tracker.append(nonTerminal)

        # Step 2: Check the rules of CNF
        for nonTerminal in self.lhs_to_rules:
            # tracker for step 3
            total_probability = 0
            for rules in self.lhs_to_rules[nonTerminal]:
                nt, rhs, prob = rules

                # in this case the result must be a terminal
                if len(rhs) == 1:
                    if rhs[0] in non_terminal_tracker:
                        return False
                # in this case the result must be two non-terminals
                elif len(rhs) == 2:
                    if (rhs[0] or rhs[1]) not in non_terminal_tracker:
                        return False

                # tracker for Step 3
                total_probability = total_probability + prob

            # Step 3: Check the total probabilities = 1 for every non-terminal
            if not math.isclose(1, total_probability):
                return False

        return True


if __name__ == "__main__":
    # with open(sys.argv[1],'r') as grammar_file:
    grammar = Pcfg(open('./atis3.pcfg', 'r'))
    if grammar.verify_grammar():
        print(True)
    else:
        raise Exception("Sorry, not a valid CNF grammar")
