# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:19:46 2020
@author: xiangguchn
"""

from collections import defaultdict
from functools import wraps

# Price of steel plate with different size
original_prica = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 35]
price = defaultdict(int)
for i,p in enumerate(original_prica):
    price[i+1] = p

# Python decorator of memory
def memo(f):
    already_computed = {}
    @wraps(f)
    def _wrap(arg):
        if arg in already_computed:
            result = already_computed[arg]
        else:
            result = f(arg)
            already_computed[arg] = result
        return result
    return _wrap

solution = {}

# return max price
@memo
def r(n):
    """
    Args: n is the iron length
    Return: the max revenue 
    """
    max_price, max_split = max(
        [(price[n], 0)] + [(r(i) + r(n-i), i) for i in range(1, n)], key=lambda x: x[0]
    )
    # save split result in solution
    solution[n] = (n - max_split, max_split)
    return max_price 


# split method
def parse_solution(n):
    # 
    left_split, right_split = solution[n]    
    # If there is only one split
    if right_split == 0: return [left_split]
    # 
    return parse_solution(left_split) + parse_solution(right_split)

    
# test if the size of steel plate n
n = 50
max_price = r(n)
split_solution = parse_solution(n)
print('For steal plate with size of ', n ,', the best split method is ',
      split_solution,', and the max price ',max_price)
