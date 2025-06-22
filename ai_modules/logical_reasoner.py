from typing import List

class LogicalReasoner:
    """Logical reasoner for hallucination detection."""
    def __init__(self):
        self.kb = []
        self.symbols = {
            'H_Google': False, 'H_Groq': False, 'H_Cohere': False,
            'S_Google': False, 'S_Groq': False, 'S_Cohere': False,
            'C_Google': False, 'C_Groq': False, 'C_Cohere': False,
            'R_Ext': True
        }
        self.inferences = []

    def add_clause(self, clause):
        self.kb.append(clause)

    def set_symbol(self, symbol, value):
        self.symbols[symbol] = value

    def forward_chaining(self):
        inferred = set()
        agenda = list(self.kb)

        while agenda:
            clause = agenda.pop(0)
            if isinstance(clause, str):
                if clause not in inferred:
                    inferred.add(clause)
                    self.symbols[clause] = True
                    for new_clause in self.kb:
                        if isinstance(new_clause, tuple) and clause in new_clause[0]:
                            agenda.append(new_clause)
            elif isinstance(clause, tuple):
                premise, conclusion = clause
                if premise in inferred and conclusion not in inferred:
                    inferred.add(conclusion)
                    self.symbols[conclusion] = True
                    agenda.extend([c for c in self.kb if isinstance(c, tuple) and conclusion in c[0]])

        if self.symbols['H_Google']:
            self.inferences.append("Google response likely hallucinated.")
        if self.symbols['H_Groq']:
            self.inferences.append("Groq response likely hallucinated.")
        if self.symbols['H_Cohere']:
            self.inferences.append("Cohere response likely hallucinated.")
        if not self.symbols['R_Ext']:
            self.inferences.append("External reference may be unreliable.")

        return self.inferences