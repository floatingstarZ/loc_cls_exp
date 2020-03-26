import numpy as np
import math
ps = [0.08, 0.2, 0.15, 0.03, 0.12, 0.02, 0.4]
s = 0
for p in ps:
    s += -math.log2(p) * p
    print(math.log2(p), p, math.log2(2))
print(s)
#
# ps = [0.08, 0.2, 0.15, 0.03, 0.12, 0.02, 0.4]
# ps = [1110, 100, 101 ,11110, 110 ,11111 ,0]
print(0.4 + (0.2+0.15+0.12)*3 + 0.08*4 + (0.03+0.02)*5)