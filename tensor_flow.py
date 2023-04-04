import numpy as np
from scipy import constants
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(constants.c)  # Speed of light
print(constants.h)  # PLanks constant
print(constants.N_A)  # Avogrado's number

# Coding challenge known as flipping bits
n = int(input('Enter your number: '))
n = str(n)
numbers = []
for i in range(31):
    n = '0' + n
    numbers.append(n)

x = numbers[30]
print(x)
list_of_numbers = [int(i) for i in x]
print(list_of_numbers)
y = len(list_of_numbers)
r = -1
list1 = []
for i in list_of_numbers:
    if i == 0:
        i = 1
    else:
        i = 0
    list1.append(i)
print(list1)
y = len(list_of_numbers)
r = -1
list2 = []
for i in range(y):
    y = y - 1
    r = r + 1
    z = 2 ** r
    t = list1[y] * z
    list2.append(t)
print(sum(list2))

# Updated version using ChatGPT
n = int(input('Enter your number: '))
bin_str = bin(n)[2:].zfill(32)  # convert to 32-bit binary string

flipped_str = ''.join(['0' if c == '1' else '1' for c in bin_str])  # flip the bits

result = int(flipped_str, 2)  # convert to decimal

print(result)
