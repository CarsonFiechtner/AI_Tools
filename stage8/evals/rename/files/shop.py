def calc_total(prices, tax=0.1):
    return round(sum(prices) * (1 + tax), 2)
