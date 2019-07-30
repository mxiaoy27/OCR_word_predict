import random
 
number = random.randint(1,10)
 
f = open('number.txt','w')
f.write(str(number))
f.close()