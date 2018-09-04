from sympy import *
def LogisticReg(x,y,m=0,b=0,lr=0.0001,epochs=10000):
    N=float(len(y))
    for i in range(epochs):
        y_current=sigmoid(m*x+b)
        cost=sum([data for data in (-1*y*(np.exp(y_current))+(1-y)*(np.exp(1-y_current)))])/N
        m_gradient=
        b_gradient=
        m=m-(learning_rate*m_gradient)
        b=b-(learning_rate*b_gradient)
    return cost,m,b
def sigmoid(x):
    return 1/(1+np.exp(-x))
