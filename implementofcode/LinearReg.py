def LinearReg(x,y,m=0,b=0,lr=0.001,epochs=10000):
    N=float(len(y))
    for i in range(epochs):
        y_current=m*x+b
        cost=sum([data**2 for data in (y-y_current)])/N
        m_gradient = -(2/N) * sum(X * (y - y_current))
        b_gradient = -(2/N) * sum(y - y_current)
        m = m - (learning_rate * m_gradient)
        b = b - (learning_rate * b_gradient)
    return m,b,cost
