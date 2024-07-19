import pstats
from pstats import SortKey

p = pstats.Stats('prof.txt')

p.sort_stats(SortKey.TOTAL).print_stats(100)
