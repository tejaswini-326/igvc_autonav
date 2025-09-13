def permutations(s):
    if len(s) == 1:
        return [s]
    all = []
    for char in s:
        all.extend([char + item for item in permutations(s.replace(char, ''))])
    return all


print(len(permutations("1234567")))