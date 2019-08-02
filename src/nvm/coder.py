import torch as tr
import itertools as it

class Coder:
    """
    Mappings between human-readable tokens and activity patterns.
    Tokens should be strings and patterns should be numpy arrays.
    patterns must be convertable to hashable type for storage in dicts.
    """

    def __init__(self, activator):
        """
        Set up a coder with a supplied pattern maker function.
        Patterns will be generated by calling activator.make_pattern.
        Patterns will be hashed by calling activator.hash_pattern.
        """
        self.activator = activator
        self.encodings = {} # maps tokens to patterns
        self.decodings = {} # maps patterns to tokens
        
    def list_tokens(self):
        """Return a list of all tokens encoded so far."""
        return self.encodings.keys()

    def encode(self, token, pattern=None):
        """
        Return the pattern encoding a token.
        If not already encoded and pattern provided, encodes using given pattern
        If not already encoded and no pattern provided, uses make_pattern()
        """
        # Encode if not already encoded
        if token not in self.encodings:
            #print(self.activator.label)
            if pattern is None:
                #print('here')
                pattern = self.activator.make_pattern()
            #print(token)
            #print(pattern.t())
            self.encodings[token] = pattern
            self.decodings[self.activator.hash_pattern(pattern)] = token
        return self.encodings[token]

    def decode(self, pattern):
        """
        Decode a pattern into a token.
        If no token has been encoded as the pattern, the default is "?"
        """
        return self.decodings.get(self.activator.hash_pattern(pattern), "?")

if __name__ == "__main__":
    
    N = 8
    PAD = 0.9
    
    from activator import *
    # act = tanh_activator(PAD, N)
    activator = logistic_activator(PAD, N)

    c = Coder(activator)
    v = c.encode("TEST")
    print(tr.transpose(v, 0,1))
    print(c.decode(v))
    print(tr.transpose(c.encode("TEST"), 0,1))
    print(c.decode(activator.make_pattern()))
