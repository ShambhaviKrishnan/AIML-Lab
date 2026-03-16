# Knowledge Base
P = True   # Example fact
implication = True   # P -> Q rule assumed

# Modus Ponens
def modus_ponens(P, implication):
    if P and implication:
        Q = True
    else:
        Q = False
    return Q

Q = modus_ponens(P, implication)
print("Result using Modus Ponens, Q =", Q)


# Resolution Example
def resolution(clause1, clause2):
    for literal in clause1:
        if "-" + literal in clause2:
            new_clause = list(set(clause1 + clause2))
            new_clause.remove(literal)
            new_clause.remove("-" + literal)
            return new_clause
    return None

clause1 = ["P", "Q"]
clause2 = ["-P", "R"]

result = resolution(clause1, clause2)
print("Resolution result:", result)


# Satisfiability Check
def check_satisfiable():
    for P in [True, False]:
        for Q in [True, False]:
            expression = (not P) or Q  # P -> Q
            if expression:
                print("Satisfiable with P =", P, "Q =", Q)
                return True
    return False

print("Satisfiability:", check_satisfiable())
