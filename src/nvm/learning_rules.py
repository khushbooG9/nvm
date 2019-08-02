import torch as tr
import numpy as np
from activator import *

def linear_solve(w, b, X, Y, actx, acty):
    dwb = totensor(np.linalg.lstsq(
        np.concatenate((fromtensor(X).T, np.ones((X.shape[1],1))), axis=1), # ones for bias
        fromtensor(acty.g(Y)).T, rcond=None)[0].T)
    dw, db =  dwb[:,:-1], dwb[:,[-1]]
    return dw, db

def hebbian(w, b, X, Y, actx, acty):
    N = X.shape[0]
    alpha = 2./(actx.on - actx.off)
    beta = (alpha * actx.off + 1)
    one = tr.ones(X.shape)
    #dw = acty.g(Y).dot(alpha**2 * X.T - alpha * beta * one.T) / N
    dw = tr.matmul(acty.g(Y), alpha**2 * tr.transpose(X,0,1) - alpha * beta * tr.transpose(one,0,1))/N
    #db = acty.g(Y).dot(- alpha * beta * X.T + beta**2 * one.T).dot(one[:,:1]) / N
    db = tr.matmul(tr.matmul(acty.g(Y),- alpha * beta * tr.transpose(X,0,1) + beta**2 * tr.transpose(one,0,1)),one[:,:1])/N
    return dw, db

def rehebbian(w, b, X, Y, actx, acty):

    N = X.shape[0]
    c = (actx.on + actx.off)/2. # center
    r = (actx.on - actx.off)/2. # radius
    if not (isinstance(w.dtype, float) and isinstance(d.dtype, float)):
                    w,b = w.float(), b.float()
    w0, b0 = w, b
    for p in range(Y.shape[1]):
        x, y = X[:,[p]], Y[:,[p]]
        dw = (acty.g(y) - (tr.matmul(w,x) +b)) * tr.transpose((x - c),0,1) / (N*r**2)
        db = totensor(- fromtensor(dw).sum(axis=1)[:,np.newaxis] * c)
        w, b = w + dw, b + db

    dw, db = w - w0, b - b0
    return dw, db

def dipole(w, b, X, Y, actx, acty):
    # only works for single x, y
    
    # map x, y to [-1,1]
    wx = 2/(actx.on - actx.off)
    bx = -(actx.on + actx.off)/(actx.on - actx.off)
    sx = tr.sign(wx*X + bx)
    wy = 2/(acty.on - acty.off)
    by = -(acty.on + acty.off)/(acty.on - acty.off)
    sy = tr.sign(wy*Y + by)
    
    # map result back to acty
    yw = (acty.g(acty.on) - acty.g(acty.off))/2
    yb = (acty.g(acty.on) + acty.g(acty.off))/2

    # final weights
    N = X.shape[0]
    one = tr.ones((N,1))
    dw = yw*sy*tr.transpose(sx,0,1)*wx - w
    #db = yw*sy*(sx.T.dot(one)*bx - (N-1)) + yb - b
    db = yw*sy*(tr.matmul(tr.transpose(sx,0,1),one))

    return dw, db

def learn(w, b, X, Y, actx, acty, learning_rule, verbose=False):
    
    if X.shape[1] > 0:

        dw, db = learning_rule(w, b, X, Y, actx, acty)
        w, b = w.double() + dw.double(), b.double() + db.double()
    
        _Y = acty.f(tr.matmul(w,X.double()) + b)
        #print("acty",acty.e(Y, _Y).dtype)
        #print("one", tr.ones(Y.shape).dtype)
        diff_count = (tr.ones(Y.shape) - acty.e(Y, _Y).float()).sum()

        # if verbose and diff_count > 0:
        if verbose:
            print("Learn residual max: %f"%tr.abs(Y - _Y).max())
            print("Learn residual mad: %f"%tr.abs(Y - _Y).mean())
            print("Learn diff count: %d"%(diff_count))

    else:

        diff_count = 0

    return w, b, diff_count


if __name__ == "__main__":
    
    # # logistic hebbian
    # N = 8
    # K = 3
    # act = logistic_activator(0.05, N)
    # X = np.empty((N,K))
    # for k in range(K):
    #     X[:,[k]] = act.make_pattern()
    
    # W, b = np.zeros((N,N)), np.zeros((N,1))
    # W, b = hebbian(W, b, X[:,:-1], X[:,1:], act, act,)
    
    # Y = act.f(W.dot(X[:,:-1]) + b)
    # print(act.e(X[:,1:], Y))

    # dipole
    N = 4
    K = 1
    act = logistic_activator(0.05, N)
    # act = tanh_activator(0.05, N)
    X = tr.empty((N,K))
    print(X.type())
    Y = tr.empty((N,K))
    for k in range(K):
        X[:,[k]] = (act.make_pattern()).float()
        Y[:,[k]] = (act.make_pattern()).float()
    
    W, b = tr.zeros((N,N)), tr.zeros((N,1))
    W, b = dipole(W, b, X, Y, act, act)
    print("Y",tr.transpose(Y,0,1))
    print("X",tr.transpose(X,0,1))
    print("W,b")
    print(W)
    print(b)
    
    Y_ = act.f(tr.matmul(W,X) + b)
    print(act.e(Y_, Y))

    idx = (tr.rand(N,1) > .5)
    print("idx",tr.transpose(idx,0,1))
    print("X",tr.transpose(X,0,1))
    for i in range(X.shape[0]):
        if idx[i]:
            if X[i,0] == act.on: X[i,0] = act.off
            else: X[i,0] = act.on
    print("X",tr.transpose(X,0,1))
    Y_ = act.f(tr.matmul(W,X) + b)
    print(act.e(Y_, Y))
