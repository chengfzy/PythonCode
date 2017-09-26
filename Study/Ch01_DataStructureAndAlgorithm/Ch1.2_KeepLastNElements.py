from collections import deque


def search(lines, pattern, history = 5):
    previous_lines = deque(maxlen = history)
    for l in lines:
        if pattern in l:
            yield l, previous_lines
        previous_lines.append(l)

# Example use on a file
if __name__ == '__main__':
    with open(r'../data/file.txt') as f:
        for line, prevlines in search(f, 'apt-get', 5):
            for pline in prevlines:
                print(pline, end=' ')
            print(line, end=' ')
            print('-' * 20)
