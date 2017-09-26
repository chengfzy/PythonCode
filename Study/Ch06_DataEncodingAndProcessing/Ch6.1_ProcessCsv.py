import csv
from collections import namedtuple

# read csv file
with open('../data/stocker.csv') as f:
    f_csv = csv.reader(f)
    headings = next(f_csv)
    Row = namedtuple('Row', headings)
    print(Row)
    for r in f_csv:
        row = Row(*r)
        print(row)

# write csv file
headers = [' Symbol ', ' Price ', ' Date ', ' Time ', ' Change ', ' Volume ']
rows = [{' Symbol ': ' AA ', ' Price ': 39.48, ' Date ': ' 6/11/2007 ',
         ' Time ': ' 9:36am ', ' Change ': -0.18, ' Volume ': 181800},
        {' Symbol ': ' AIG ', ' Price ': 71.38, ' Date ': ' 6/11/2007 ',
         ' Time ': ' 9:36am ', ' Change ': -0.15, ' Volume ': 195500},
        {' Symbol ': ' AXP ', ' Price ': 62.58, ' Date ': ' 6/11/2007 ',
         ' Time ': ' 9:36am ', ' Change ': -0.46, ' Volume ': 935000},
        ]


with open('../data/stocksB.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

