import ctf
import reader
T = reader.read_from_frostt('nell-2', 12092, 9184, 28818)
assert(T[0,182,606] == 1.0)
