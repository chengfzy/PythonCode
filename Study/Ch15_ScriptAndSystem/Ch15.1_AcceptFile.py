import fileinput

print("Hello")
with fileinput.input() as f_input:
    for line in f_input:
        print(line, end='')

print("End")
