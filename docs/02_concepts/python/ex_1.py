# 1. Write a program to prompt a user to enter his/her name and mailing address and display the entered information on the screen
# name = input('Please write your name')
# add = input('Please enter your email id')
# print(f"thank you for entering your name {name}")
# print(f"your email id: {add}")

# # 2. Write a program to compute the area of different shapes and display the info on the screen
# import math
# radius = float(input('Enter the radius of the circle')) # have to typecast to float here
# print(f"The area of the circle is {float(math.pi*radius*radius)}")

# #3. Write a program to compute the factorial of a given number and display the result on the screen
# num = int(input('Enter the number: \n'))
# fact = 1
# for i in range(2, num+1):
#     fact = fact * i
# print(fact)

# # 4. Write a program to check if a given number is prime and display the result on the screen
# num = int(input('Enter the number: \n'))
# import math
# prime = True
# x = int(math.sqrt(num))
# while x>1:
#     if num%x==0:
#         print(x, 'divides', num)
#         prime = False
#         break
#     else:
#         x=-1
# if prime:
#     print('Prime')
# else:
#     print('no')

# 5. Write a program to find the GCD of two numbers
num = input