#vectorization

import numpy as np
import time as time



# n1=np.random.rand(10000000)
# n2=np.random.rand(10000000)
# tic=time.time()
# n3=np.dot(n1,n2)
# toc=time.time()
# #creates 10 values between 0 and 1
# usingVector=0
# usingVector=toc-tic
# print("Using Vector : " + str(usingVector))

# tic=time.time()
# op=0
# for i in range(10000000):
#     op=op+n1[i]*n2[i]
# toc=time.time()

# loop=toc-tic

# print("Using Loops: " + str(loop))

# print(loop/usingVector)

a=np.random.rand(10)
u=np.exp(a)
print(a)
print(u)
print(np.log(u))
print("Maximum : "+str(np.max(a,0)))
print("Maximum : "+str(np.max(u,0)))
