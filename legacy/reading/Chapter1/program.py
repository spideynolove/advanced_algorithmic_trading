comparisons = [(i, j) for i in range(6) for j in range(i+1, 6)]
pos = [(k, v) for k in range(5) for v in (0, 1)]
tmp = zip(comparisons, pos)
# print(len(tmp))
for item in tmp:
    print(item)
# ((0, 1), (0, 0))
# ((0, 2), (0, 1))
# ((0, 3), (1, 0))
# ((0, 4), (1, 1))
# ((0, 5), (2, 0))
# ((1, 2), (2, 1))
# ((1, 3), (3, 0))
# ((1, 4), (3, 1))
# ((1, 5), (4, 0))
# ((2, 3), (4, 1))
