import sys
sys.path.append('../nvm')

from random import randint, choice, sample, gauss
from math import sqrt, asin
from itertools import chain

from layer import Layer
from activator import *
from coder import Coder
from learning_rules import rehebbian

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from test_abduction import test_data as abduction_test_data
from test_abduction import build_fsm, abduce


class GraphNet:
    def __init__(self, N, mask_frac, stabil=10):
        self.stabil = stabil
        self.mask_frac = mask_frac

        size = N**2
        self.size = size
        self.mask_size = int(self.size / self.mask_frac)
        pad = 0.0001

        ### Create layers
        self.act = tanh_activator(pad, size)
        #self.act_mask = gate_activator(pad, size)
        #self.act_mask = logistic_activator(pad, size)
        self.act_mask = heaviside_activator(size)

        self.reg_layer, self.mem_layer, self.ptr_layer = (
            Layer(k, (N,N), self.act, Coder(self.act)) for k in "rmp")

        # Gating masks
        self.masks = { }
        self.w_mask = np.zeros((size,size))

        # Weight matrices
        self.w_mm = np.zeros((size,size))
        self.w_pm = np.zeros((size,size))
        self.w_mp = np.zeros((size,size))

        # Dummy bias to avoid extra memory allocation
        self.dummy_bias = np.zeros((size, 1))


    def learn(self, mappings):
        kfts = [(k,start,v) for start,m in mappings.items() for k,v in m]
        key_syms, from_syms, to_syms = zip(*tuple(kfts))
        mem_syms = list(set(from_syms + to_syms))
        uniq_keys = list(set(key_syms))

        # Learn all memory attractors and mem->ptr transfers
        X_mem = self.mem_layer.encode_tokens(mem_syms)
        X_ptr = self.ptr_layer.encode_tokens(mem_syms)
        self.w_mm += rehebbian(self.w_mm, self.dummy_bias, X_mem, X_mem, self.act, self.act)[0]
        self.w_pm += rehebbian(self.w_pm, self.dummy_bias, X_mem, X_ptr, self.act, self.act)[0]

        # Construct and learn masks
        Y_masks = np.zeros((self.size,len(uniq_keys)))
        for i in range(len(uniq_keys)):
            Y_masks[np.random.choice(
                self.size, self.mask_size, replace=False),i] = 1.
        X_reg = self.reg_layer.encode_tokens(uniq_keys)
        self.w_mask += rehebbian(self.w_mask, self.dummy_bias, X_reg, Y_masks, self.act, self.act_mask)[0]

        # Reconstruct randomly generated masks
        self.masks = {
            k : self.act_mask.f(
                    self.w_mask.dot(
                        self.reg_layer.coder.encode(k)))
            for k in uniq_keys
        }

        # Relearn from/to ptr/mem attractors
        X = self.ptr_layer.encode_tokens(from_syms)
        Y = self.mem_layer.encode_tokens(to_syms)
        self.w_mm += rehebbian(self.w_mm, self.dummy_bias, Y, Y, self.act, self.act)[0]

        # Learn from/to ptr/mem transitions
        all_masks = np.concatenate(tuple(self.masks[k] for k in key_syms), axis=1)
        X = np.multiply(X, all_masks)
        Y = np.multiply(Y, all_masks)
        self.w_mp += rehebbian(self.w_mp, self.dummy_bias, X, Y, self.act, self.act)[0]




    def test_recovery(self, mappings):
        # Accuracy of Hebbian mem recovery
        w_correct = 0.
        # Symbolic final correct
        correct = 0
        # Total tests
        total = 0

        mem_states = set(k for k in mappings.keys()).union(
            set(v for m in mappings.values() for k,v in m))

        #print("--Testing recovery:")
        #print("--  mask_frac = %d" % self.mask_frac)
        #print("--  num_masks = %d" % len(self.masks))
        #print("--  mem_states = %d" % len(mem_states))
        #print("--  total = %d" % (len(self.masks) * len(mem_states)))
        for tok in mem_states:
            complete = self.mem_layer.coder.encode(tok)

            for mask in self.masks.values():
                partial = np.multiply(complete, mask)
                y_mem = self.act.f(self.w_mm.dot(partial))

                # Stabilize
                for _ in range(self.stabil):
                    y_mem = self.act.f(self.w_mm.dot(y_mem))
                out = self.mem_layer.coder.decode(y_mem)

                # Check output
                if out == tok:
                    w_correct += 1
                    correct += 1
                else:
                    w_correct += (
                        np.sum(np.sign(y_mem) == np.sign(complete))
                        / y_mem.size)
                total += 1

        return float(correct) / total, w_correct / total

    def test_traversal(self, mappings):
        # Accuracy of transfer from mem->ptr
        p_correct = 0.
        # Accuracy of gated transition from ptr->mem
        t_correct = 0.
        # Accuracy of Hebbian mem recovery
        w_correct = 0.
        # Symbolic final correct
        correct = 0
        # Total tests
        total = 0

        input_keys = set(k for m in mappings.values() for k,v in m)

        #print("--Testing traversal:")
        #print("--  num_inputs = %d" % (len(input_keys)))
        #print("--  num_transits = %d" % sum(len(m) for m in mappings.values()))

        for start,m in mappings.items():
            start_pat = self.mem_layer.coder.encode(start)
            ptr_target = self.ptr_layer.coder.encode(start)

            for inp,end in m:
                mask = self.masks[inp]
                mem_target = self.mem_layer.coder.encode(end)

                # Compute ptr activation
                y_ptr = np.sign(self.act.f(self.w_pm.dot(start_pat)))
                p_correct += (np.sum(y_ptr == np.sign(ptr_target)) / y_ptr.size)

                # Mask and transit
                y_ptr = np.multiply(y_ptr, mask)
                y_mem = np.multiply(self.act.f(self.w_mp.dot(y_ptr)), mask)

                t_correct += 1 - (
                    np.sum(np.multiply(mask, np.abs(y_mem - mem_target) / 2))
                    / self.mask_size)

                # Stabilize mem
                for _ in range(self.stabil):
                    y_mem = self.act.f(self.w_mm.dot(y_mem))
                out = self.mem_layer.coder.decode(y_mem)

                # Check output
                if out == end:
                    w_correct += 1
                    correct += 1
                else:
                    w_correct += (
                        np.sum(np.sign(y_mem) == np.sign(mem_target))
                        / y_mem.size)
                total += 1

        return (float(correct) / total,
            w_correct / total,
            t_correct / total,
            p_correct / total)

    def test(self, mappings):
        return {
            "trans_acc" : self.test_traversal(mappings),
            "recall_acc" : self.test_recovery(mappings) }

def print_results(prefix, results):
    print(
        ("%7s" % prefix) +
        " ".join("      " + " / ".join("%6.4f" % r for r in results[k])
            for k in [
                "trans_acc", "recall_acc" ]))



def gen_mappings(num_states, num_inputs, num_trans):
    # Create finite state machine with input conditional transitions
    mem_states = [str(x) for x in range(num_states)]
    input_tokens = [str(x) for x in range(num_inputs)]

    # Encode transitions
    mappings = dict()
    for f in mem_states:
        others = [x for x in mem_states if x != f]
        s = np.random.choice(others, num_trans, replace=False)
        t = np.random.choice(input_tokens, num_trans, replace=False)
        mappings[f] = list(zip(t,s))

    return mappings

def table_to_mappings(table):
    return {
        str(i): [(table[i][j], str(j))
                    for j in range(len(table))
                        if table[i][j] is not None]
        for i in range(len(table))
    }

def test(N, mask_frac, mappings):
    n = GraphNet(N, mask_frac)
    n.learn(mappings)
    return n.test(mappings)

def print_header():
    print("                   " + " ".join(
        "%21s" % x for x in [
        "final_acc         trans_acc", "recall_acc"]))

def test_random_networks(N, mask_frac):
    print_header()

    num_nodes = N * 2
    for p in [0.1, 0.5]:
        net = nx.fast_gnp_random_graph(num_nodes, p, directed=True)
        #print(len(net.nodes), len(net.edges))

        edges = {}
        for u,v in net.edges:
            if u not in edges: edges[u] = []
            edges[u].append(v)

        mappings = {}
        keys = range(max(len(vs) for vs in edges.values()))
        for u,vs in edges.items():
            if u not in mappings: mappings[u] = [(i,v)
                for i,v in zip(sample(keys,len(vs)), vs)]
                #for i,v in enumerate(vs)]

        n = GraphNet(N, mask_frac)
        n.learn(mappings)
        print_results("%d/%d" % (num_nodes, len(net.edges)), n.test(mappings))
    print("")

def test_machines(N, mask_frac):
    print_header()

    table = [
        [None, "A", "B", "C", "D", "E", "F", "G"],
        ["D", None, "A", None, None, None, None, None],
        ["D", None, None, "B", None, "C", None, None],
        ["D", None, None, None, None, None, None, None],
        ["D", None, "C", None, "A", None, None, "B"],
        ["D", None, None, None, None, None, None, None],
        ["D", None, "A", "B", "C", None, None, None],
        ["D", None, None, None, "B", None, "A", None],
    ]

    mappings = table_to_mappings(table)
    print_results("Machine A", test(N, mask_frac, mappings))

    mappings = {
        "1" : [("A", "2"), ("B", "3"), ("C", "4"), ("D", "6")],
        "2" : [("A", "1")],
        "3" : [("B", "1")],
        "4" : [("A", "5")],
        "5" : [("C", "1")],
        "6" : [("A", "7"), ("B", "8"), ("C", "9")],
        "7" : [("B", "1")],
        "8" : [("A", "2")],
        "9" : [("A", "3")],
    }

    print_results("Machine B", test(N, mask_frac, mappings))
    print("")

def test_param_explore(N, mask_frac):
    num_states = N * 2
    num_inputs = int(N ** 0.5)
    num_trans = int(N ** 0.5)

    print("num_states = %d" % num_states)
    print("num_inputs = %d" % num_inputs)
    print("num_trans = %d" % num_trans)
    print("")

    print_header()

    print("mask_frac")
    for x in [mask_frac, mask_frac ** 2]:
        mappings = gen_mappings(num_states, num_inputs, num_trans)
        print_results(x, test(N, x, mappings))
    print("")

    print("num_states")
    for x in [N*2, N * 4]:
        x = int(x)
        mappings = gen_mappings(x, num_inputs, num_trans)
        print_results(x, test(N, mask_frac, mappings))
    print("")

    print("num_inputs")
    for x in [N * 4, N * 8]:
        x = int(x)
        mappings = gen_mappings(num_states, x, num_trans)
        print_results(x, test(N, mask_frac, mappings))
    print("")

    print("num_trans")
    for x in [N, N*2]:
        x = int(x)
        mappings = gen_mappings(max(num_states,x+1), max(num_inputs,x), x)
        print_results(x, test(N, mask_frac, mappings))
    print("")

def test_traj(N, mask_frac):
    # Create field
    field_states = [("f%d" % i) for i in range(N * 2)]

    # Create heads
    heads = [("h%d" % i) for i in range(N)]
    traj_length = N

    print("field_states = %d" % len(field_states))
    print("heads        = %d" % len(heads))
    print("traj_length  = %d" % traj_length)
    print("")

    trajs = []
    pairs = []
    mappings = {}

    # Chain states together using head as key
    for h in heads:
        traj = np.random.choice(field_states, traj_length, replace=False)
        trajs.append(traj)

        for i in range(traj_length-1):
            pairs.append((traj[i], h, traj[i+1]))
        pairs.append((traj[-1], h, "NULL"))

    for pre,h,post in pairs:
        if pre not in mappings:
            mappings[pre] = [ (h, post) ]
        else:
            mappings[pre].append((h, post))

    print_header()
    #for k,v in mappings.items(): print(k,v)
    print_results("Traj", test(N, mask_frac, mappings))

    # Use numerical indices from head to each state
    mappings = {}
    for h,traj in zip(heads,trajs):
        if h not in mappings:
            mappings[h] = []
        mappings[h] = list(enumerate(traj))

    #for k,v in mappings.items(): print(k,v)
    print_results("Indexed", test(N, mask_frac, mappings))

    # Use states as index into head
    mappings = {}
    for pre,h,post in pairs:
        if h not in mappings:
            mappings[h] = [ (pre, post) ]
        else:
            mappings[h].append((pre, post))

    #for k,v in mappings.items(): print(k,v)
    print_results("Rev-ndx", test(N, mask_frac, mappings))
    print("")

def test_abduce():
    test_data = []
    actions = 'ABC'
    causes = 'XYZSTUGHIJ'
    symbols = actions + causes

    p = [1. / len(actions) for a in actions]
    p = [x /sum(p) for x in p]

    knowledge = []
    for i in range(12):
        knowledge.append((
            sample(causes,1)[0] , tuple(
                symbols[i] for i in np.random.choice(len(p),2, p=p))))

    p = [.5 / len(actions) for a in actions] + [.5 / len(causes) for c in causes]
    p = [x /sum(p) for x in p]
    for i in range(8):
        knowledge.append((
            sample(causes,1)[0] , tuple(
                symbols[i] for i in np.random.choice(len(p),2, p=p))))

    print("Knowledge:")
    for k in knowledge:
        print("  " + str(k))

    seq = "".join([choice('ABC') for _ in range(256)])
    for l in (2, 4, 8, 16, 32):
        test_data.append(
            (
                knowledge,
                seq[:l],
                None
            ))

    for knowledge, seq, answer in test_data:
    #for knowledge, seq, answer in abduction_test_data:
        fsm = build_fsm(knowledge)
        timepoints, best_path = abduce(fsm, seq)
        fsm_states = list(iter(fsm))
        causes = [c for t in timepoints for c in t.causes]

        data = {
            #"curr_t" : str(timepoints[0]),
            #"fsm" : str(fsm),
        }

        data.update({
            str(s) : {
                'parent' : str(s.parent),
                'transitions' : [ inp for inp in s.transitions ],
                'causes' : [ str(c) for c in s.causes ],
            } for s in fsm_states
        })
        for s in fsm_states:
            data[str(s)].update({
                inp : str(s2) for inp,s2 in s.transitions.items() })

        data.update({
            str(t) : {
                'previous' : str(t.previous),
                'causes' : [ str(c) for c in t.causes ],
            } for t in timepoints
        })

        data.update({
            str(c) : {
                'identity' : c.identity,
                'start_t' : str(c.start_t),
                'end_t' : str(c.end_t),
                'source_state' : str(c.source_state),
                'cache' : [ str(s) for s in c.cache ],
                'effects' : [ str(e) for e in c.effects ],
            } for c in causes
        })
        for c in causes:
            data[str(c)].update({
                str(s) : str(c) for s,c in c.cache.items() })

        #for x in [str(x) for x in fsm_states + timepoints + causes]:
        #    print(x, data[x])

        pairs = []

        for parent,d in data.items():
            if type(d) is dict:
                for key,value in d.items():
                    if type(value) is list:
                        if len(value) > 0:
                            # Chain data together
                            pairs.append((parent, key, value[0]))
                            for (v1,v2) in zip(value, value[1:]):
                                pairs.append((v1,parent,v2))
                            pairs.append((value[-1], parent, "None"))
                        else:
                            pairs.append((parent, key, "None"))
                    else:
                        pairs.append((parent, key, value))

        mappings = {}
        for pre,key,post in pairs:
            if pre not in mappings:
                mappings[pre] = [ (key, post) ]
            else:
                mappings[pre].append((key, post))
        print(seq)
        #for k in sorted(mappings.keys()):
        #    print(k, mappings[k])

        mem_states = set()
        reg_states = set()
        for k,v in mappings.items():
            mem_states.add(k)
            for inp,res in v:
                reg_states.add(inp)
                mem_states.add(res)
        reg_states = reg_states - set(mem_states)
        print("Timepoints: ", len(timepoints))
        print("FSM states: ", len(fsm_states))
        print("Causes: ", len(causes))
        print("Memory states: ", len(mem_states))
        print("Register states: ", len(reg_states))
        print("Total states: ", len(mem_states.union(reg_states)))
        print("Total transitions: ", len(pairs))

        print()
        print_header()
        for N in [16, 24, 32]:
            print_results(N, test(N, (N**0.5)/2, mappings))
        print()

# Parameters
for N in [16, 24, 32]:
    mask_frac = (N ** 0.5) / 2

    print("-" * 80)
    print("N=%d" % N)
    print("mask_frac = %s" % mask_frac)
    print()

    test_random_networks(N, mask_frac)
    test_machines(N, mask_frac)
    test_param_explore(N, mask_frac)
    test_traj(N, mask_frac)
print("-" * 80)
test_abduce()
