### newmodules
import sympy
from sympy import I as Im
from sympy import conjugate
from sympy import sympify, symbols, simplify, Symbol, Mul, expand
from sympy.parsing.latex import parse_latex
import antlr4

## Utils

# Function to search for a symbol in a expression
from sympy import symbols, sqrt, assuming, Q

def find_symbol(expr, target: str):
    symb_lst = list(expr.free_symbols)
    names_lst = []
    if len(symb_lst) > 0: 
        for symbol in expr.free_symbols:
            
            names_lst.append(symbol.name)

            if symbol.name in target:
                return symbol
            
            else:
                pass
        
        # print(f"Symbol {target} not found. \n The symbols in the expression are {expr.free_symbols}")
                #pass
        # print(f"Symbol {target} not found. \n The symbols in the expression are {expr.free_symbols}")
        # return None
        return symbols(target, real=True, positive=True)
    # symbol not found, create the same
    else:
        return symbols(target, real=True, positive=True)


# Unify the symbols of different expressions
def unify_symbols(expr1, expr2):
    symbols_expr1 = set([x.name for x in expr1.free_symbols])
    symbols_expr2 = set([x.name for x in expr2.free_symbols])

    common_symbols = symbols_expr1.intersection(symbols_expr2)
    #print(common_symbols)

    symbol_mapping = {}
    for symbol in common_symbols:
        new_symbol = Symbol(symbol)
        symbol_mapping[symbol] = new_symbol

    #print(symbol_mapping)
    expr1 = expr1.subs(symbol_mapping)
    expr2 = expr2.subs(symbol_mapping)

    return expr1, expr2

# Define LaTeX expressions
# Parse the LaTeX expressions - EXAMPLE
# Unify symbols with the same text names
#unified_expr1, unified_expr2 = unify_symbols(expr1, expr2)
#dl.replace(dl.args[1].args[0], n)
#print("Unified Expression 1:", unified_expr1)
#print("Unified Expression 2:", unified_expr2)

# Unify the symbols of different expressions


## Eq 2.36 
def somm_corr():

    expr1 = r"\frac{\pi x}{1-e^{-\pi x}}"
    expr2 = r"\prod_{b = 1}^{l} (1 + \frac{x^2}{4b^2} )"
    
    fact1 = parse_latex(expr1)
    fact2 = parse_latex(expr2)


    result = fact1*fact2

    return result




def dln(np):
    expr = []
    expr.append(r"\frac{n! (2l+2n+1)!}{(l+n)!}")
    expr.append(r"\sum_{j = 0}^{2n} \frac{(-2)^j (l+j)!}{j! (2n-j)! (2l+j+1)!}")
    expr.append(r"\left[\prod_{b = l+1}^{l+j}\left(1 + \frac{i x}{2b}\right)\right]")
    
    fact = []
    result = 1
    
    
    for i, exp in enumerate(expr):
        fact.append(parse_latex(exp))
        result *= fact[i]

    nnew = Symbol(f'{np}')
    result = result.replace(find_symbol(result, 'n'), nnew)
    return result