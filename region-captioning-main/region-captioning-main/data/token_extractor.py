i_file = open("captions.txt", 'r')
o_file = open("tokens.txt", 'w')
lines = i_file.readlines()

for i in range(1,len(lines)):
    line = lines[i].split(',')[1]
    o_file.write(line)

i_file.close()
o_file.close()
