with open('results1.txt') as f:
    lines1 = f.readlines()

with open('results2.txt') as f:
    lines2 = f.readlines()

assert len(lines1) == len(lines2)
count = 0

for i in range(len(lines1)):
    if (lines1[i] != lines2[i]):
        count += 1
        print("ERROR:", i, lines1[i], lines2[i])

print ((count/len(lines1)) * 100)