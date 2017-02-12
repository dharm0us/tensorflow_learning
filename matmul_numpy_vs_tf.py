import numpy
import tensorflow as tf
import time

def getTestData():
    A = [[1., 2., 3., 4.],[3.,4.,5.,6.],[7.,8,9.,10.],[11.,12.,13.,14.]]
    return 3,A

def tfMatMul():
    n,A = getTestData()
    A = tf.constant(A)
    sess = tf.Session()
    for num in range(1,n):
        A = tf.matmul(A,A)
    output = sess.run(A)
    sess.close()
    return output

def numPyMatMul():
    n,A = getTestData()
    for num in range(1,n):
        A = numpy.matmul(A,A)
    return A


def timedRun(methodToRun):
    start = time.time()
    result = methodToRun()
    end = time.time()
    diff = end - start
    print("Method: "+str(methodToRun)+"Time Taken :"+str(diff))
    print(result)

timedRun(numPyMatMul)
timedRun(tfMatMul)


