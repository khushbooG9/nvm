import numpy as np
import torch as tr
from layer import Layer
from coder import Coder
from tensor import *

class Sequencer(object):

    def __init__(self, sequence_layer, input_layers):
        self.sequence_layer = sequence_layer
        self.input_layers = input_layers
        self.transits = []

    def add_transit(self, new_state=None, **input_states):

        # Generate states if not provided, encode as necessary
        if new_state is None:
            new_state = self.sequence_layer.activator.make_pattern()
        if type(new_state) is str:
            new_state = self.sequence_layer.coder.encode(new_state)

        for name, pattern in input_states.items():
            if type(pattern) is str:
                input_states[name] = self.input_layers[name].coder.encode(pattern)

        # Check for non-determinism
        for n, i in self.transits:
            # Same new state
            if self.sequence_layer.activator.e(n, new_state).all(): continue
            # Different input layers
            if set(i.keys()) != set(input_states.keys()): continue
            # Different input patterns
            #if any((i[l] != p).any() for l,p in input_states.items() ): continue
            find = []
            for l,p in input_states.items():
                # if not isinstance(i[l], torch.Tensor):
                #     i[l] = totensor(i[l]).double()
                # if not isinstance(p, torch.Tensor):
                #     p = totensor(p).double()
                p = totensor(p).double()
                i[l] = totensor(i[l]).double()
                if (i[l]!=p).any():
                    find.append(True)
                else:
                    find.append(False)
            if any(find): continue

            # Otherwise non-deterministic
            raise Exception("Created non-deterministic transit!")

        # Save transit
        self.transits.append((new_state, input_states))

        # Return new state
        return new_state
        
    def flash(self, verbose=False):

        # Unzip transits
        all_new_states, all_input_states = zip(*self.transits)
        P = len(self.transits)

        # Populate input matrices
        X = {}
        for i, input_states in enumerate(all_input_states):
            for name, pattern in input_states.items():
                if name not in X: X[name] = tr.zeros((pattern.shape[0]+1, P))
                X[name][:-1, [i]] = pattern.float()
                X[name][-1, i] = 1. # bias

        # Fix layer order, make sure sequence layer comes first for zsolve
        # explicitly convert to list for python3
        names = list(X.keys())
        names.remove(self.sequence_layer.name)
        names.insert(0, self.sequence_layer.name)
        
        # Solve with hidden step
        X = tr.cat([X[name] for name in names], dim=0)
        Y = tr.cat(all_new_states, dim=1)
        W, Z, residual = zsolve(X, Y,
            self.sequence_layer.activator.f,
            self.sequence_layer.activator.g,
            verbose=verbose)
        
        # Split up weights and biases
        weights = {}
        biases = {}
        offset = 0
        for name in names:
            pair_key = (self.sequence_layer.name, name)
            layer_size = self.input_layers[name].size
            weights[pair_key] = W[:,offset:offset + layer_size]
            biases[pair_key] = W[:,[offset + layer_size]]
            offset += layer_size + 1
        
        # return final weights, bias, matrices, residual
        return weights, biases, (X, Y, Z), residual

def zsolve(X, Y, f, g, verbose=False):
    """
    Construct W that transitions states in X to corresponding states in Y
    X, Y are arrays, with paired activity patterns as columns
    f, g are the activation function and its inverse    
    To deal with low-rank X, each transition uses an intermediate "hidden step"
    """

    # size of layer being sequenced
    N = Y.shape[0]
    
    # for low-rank X, get coefficients A of X's column space
    _, sv, A = tr.svd(X)
    rank_tol = tr.max(sv) * max(X.shape) * tr.finfo(sv.dtype).eps # from numpy
    A = A[sv > rank_tol, :]
    
    # use A to set intermediate Z that is low-rank pre non-linearity
    Z = tr.zeros(X.shape)
    Z[:N,:] = f(tr.matmul(tr.randn(N, A.shape[0]),A))
    Z[N,:] = 1. # bias

    # solve linear equations
    XZ = tr.cat((X, Z), dim=1)
    ZY = tr.cat((Z[:N,:], Y), dim=1)
    W = totensor(np.linalg.lstsq(fromtensor(tr.transpose(XZ,0,1)), fromtensor(tr.transpose(g(ZY),0,1)), rcond=None)[0].T)

    residual = tr.abs(ZY - f(tr.matmul(W, XZ))).max()
    if verbose: print("Sequencer flash residual = %f"%residual)

    # solution and hidden patterns
    return W, Z, residual

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, formatter = {'float': lambda x: '% .2f'%x})

    N = 8
    PAD = 0.05

    from activator import *
    act = tanh_activator(PAD, N)
    # act = logistic_activator(PAD, N)

    c = Coder(act)
    
    g = Layer("gates",[N], act, c)
    input_layers = {name: Layer(name, [N], act, c) for name in ["gates","op1","op2"]}
    s = Sequencer(g, input_layers)
    v_old = g.coder.encode("SET") # s.add_transit(new_state="SET")
    for to_layer in ["FEF","SC"]:
        for from_layer in ["FEF","SC"]:
            v_new = s.add_transit(
                new_state = to_layer + from_layer,
                gates = v_old, op1 = to_layer, op2 = from_layer)

    print(c.list_tokens())

    weights, biases, _, residual = s.flash()
    for k in weights:
        w, b = weights[k], biases[k]
        print(k)
        print(w)
        print(tr.transpose(b,0,1))

    a = {"gates":v_old, "op1":c.encode("SC"), "op2":c.encode("SC")}
    wvb = tr.zeros(v_old.shape)
    for k in weights:
        w, b = weights[k], biases[k]
        wvb += tr.matmul(w, a[k[1]]) + b
    z = tr.zeros(v_old.shape)
    a = {"gates":act.f(wvb), "op1": z, "op2":z}
    wvb = tr.zeros(v_old.shape)
    for k in weights:
        w, b = weights[k], biases[k]
        wvb += tr.matmul(w, a[k[1]]) + b
    v_test = act.f(wvb)

    for v in [v_old, v_test, v_new]:
        print(c.decode(v), tr.transpose(v,0,1))
    print(tr.transpose(act.e(v_test, v_new),0,1))
