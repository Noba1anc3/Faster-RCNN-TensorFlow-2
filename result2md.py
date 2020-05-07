
with open("../1.md", 'r') as f:
    lines = f.readlines()
    i = 1
    newLine = ''

    ap = []
    ar = []

    for index in range(len(lines)):
        line = lines[index]
        if i == 13:
            i = 1
            ar.append(newLine)
            newLine = ''
            continue
        line = line.split("]")[1].split("=")[1].split("\n")[0].replace(" ", "")
        line = round(float(line)*100, 1)
        line = str(line)
        if i == 1 or i == 2 or i == 3:
            if len(line) == 1:
                newLine += '   ' + line + ' |'
            if len(line) == 3:
                newLine += '  ' + line + ' |'
            if len(line) == 4:
                newLine += ' ' + line + ' |'
        if i == 4:
            newLine += ' nan |'
        if i == 5 or i == 6:
            if len(line) == 1:
                newLine += '     ' + line + ' |'
            if len(line) == 3:
                newLine += '   ' + line + ' |'
            if len(line) == 4:
                newLine += '  ' + line + ' |'
        if i == 7:
            ap.append(newLine)
            newLine = ''
        if i == 7 or i == 8 or i == 11 or i == 12:
            if len(line) == 1:
                newLine += '   ' + line + ' |'
            if len(line) == 3:
                newLine += '  ' + line + ' |'
            if len(line) == 4:
                newLine += ' ' + line + ' |'
        if i == 10:
            newLine += ' nan |'
        if i == 9:
            if len(line) == 1:
                newLine += '     ' + line + ' |'
            if len(line) == 3:
                newLine += '   ' + line + ' |'
            if len(line) == 4:
                newLine += '  ' + line + ' |'
        i += 1
        if index + 1 == len(lines):
            ar.append(newLine)

    for index in range(len(ap)):
        item = ap[index]
        # item = item.split("|")[0].replace(" ","")
        # print(item)
        if index == 0:
            print('|   ' + str(500) + ' |' + item)
        elif index >= 19:
            print('| ' + str((index+1) * 500) + ' |' + item)
        else:
            print('|  ' + str((index+1) * 500) + ' |' + item)

    print('\n')

    for index in range(len(ar)):
        item = ar[index]
        if index == 0:
            print('|   ' + str(500) + ' |' + item)
        elif index >= 19:
            print('| ' + str((index+1) * 500) + ' |' + item)
        else:
            print('|  ' + str((index+1) * 500) + ' |' + item)
