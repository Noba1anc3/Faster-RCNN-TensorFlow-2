def ap_ar_analysis():
    with open("../../2.md", 'r') as f:
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
            line = round(float(line) * 100, 1)
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
                print('| ' + str((index + 1) * 500) + ' |' + item)
            else:
                print('|  ' + str((index + 1) * 500) + ' |' + item)

        print('\n')

        for index in range(len(ar)):
            item = ar[index]
            if index == 0:
                print('|   ' + str(500) + ' |' + item)
            elif index >= 19:
                print('| ' + str((index + 1) * 500) + ' |' + item)
            else:
                print('|  ' + str((index + 1) * 500) + ' |' + item)


def loss_analysis():
    with open("../../all3.mdown", 'r') as f:
        lines = f.readlines()

        loss = []
        rpn_cls = []
        rpn_bbox = []
        rcnn_cls = []
        rcnn_bbox = []

        for index in range(len(lines)):
            line = lines[index]
            loss.append(line.split("Loss:")[1].split("rpn")[0])
            rpn_cls.append(line.split("Class Loss:")[1].split("RPN")[0])
            rpn_bbox.append(line.split("Bbox Loss:")[1].split("RCNN")[0])
            rcnn_cls.append(line.split("RCNN Class Loss:")[1].split("RCNN")[0])
            rcnn_bbox.append(line.split("RCNN Bbox Loss:")[1].split("\n")[0])

        # for index in range(len(lines)):
        #     print(loss[index])

        # for index in range(len(lines)):
        #     print(rpn_cls[index])

        # for index in range(len(lines)):
        #     print(rpn_bbox[index])

        # for index in range(len(lines)):
        #     print(rcnn_cls[index])

        # for index in range(len(lines)):
        #     print(rcnn_bbox[index])


# ap_ar_analysis()

loss_analysis()
