#
# Copyright John Reid 2013, 2014
#

"""
A python package for non-parametric sequence models.
"""


import logging
_logger = logging.getLogger(__name__)

import numpy.random
import seqan
from copy import copy
from collections import defaultdict


# Types we use to create strings and indexes
Value = seqan.DNA
uniformovervalues = 1. / Value.valueSize  # uniform distribution over all the values
String = seqan.StringDNA
StringSet = seqan.StringDNASet
ESA = seqan.IndexStringDNASetESA


def quote(s):
    """Wrap the string in quotes."""
    return '"%s"' % s


def prefixfor(it):
    """The prefix for a prefix tree iterator is the reverse of
    its representative."""
    return str(it.representative)[::-1]


def make_prefix_index(seqs):
    "Make an index out of the reverse of the sequences."
    sequences = StringSet()
    for seq in seqs:
        _logger.info('Building prefix index from: %s', seq)
        sequences.appendValue(String(seq[::-1]))
    return ESA(sequences)


def count_prefixes(prefix_tree, prefix_counts=None, i=None):
    """Count how many times each prefix occurs in the prefix_tree.
    """
    from itertools import imap
    if prefix_counts is None:
        prefix_counts = dict()
    if i is None:
        i = prefix_tree.topdown()
    # Check all occurrences match
    assert [i.representative] * i.numOccurrences == \
        [prefix_tree.text[occ.i1][occ.i2:occ.i2+i.repLength]
            for occ in i.occurrences]
    # Count how many occurrences match the whole string
    prefix_count = sum(imap(
        lambda occ: occ.i2 + i.repLength == len(prefix_tree.text[occ.i1]),
        i.occurrences))
    if prefix_count:
        prefix_counts[copy(i)] = prefix_count
        _logger.debug('Have %3d prefixes of: "%s"',
                      prefix_count, str(i.representative)[::-1])

    if i.goDown():
        while True:
            count_prefixes(prefix_tree, prefix_counts, copy(i))
            if not i.goRight():
                break

    return prefix_counts


def count_contexts(prefix_tree, prefix_counts):
    """Create a dictionary mapping vertexes to counts. Each set of
    counts for a vertex reflects the number of times those bases follow
    the context that the vertex represents.
    """
    #context_counts = defaultdict(lambda: numpy.zeros(Value.valueSize, dtype=int))
    context_counts = numpy.zeros((2 * len(prefix_tree), Value.valueSize), dtype=int)
    for prefix_i, count in prefix_counts.iteritems():
        # context is all but last symbol, reversed
        prefix = str(prefix_i.representative)[::-1]
        u = prefix[:-1]
        #x = prefix_i.representative[prefix_i.repLength-1]
        x = prefix_i.representative[0]
        #_logger.debug('prefix = "%s"', prefix)
        #_logger.debug('u      =  %s', u)
        #_logger.debug('x      =  %s%s', ' ' * len(u), x)
        assert prefix == u + str(x)
        #_logger.debug(u[::-1])
        #_logger.debug(str(prefix_i.representative)[1:])
        assert u[::-1] == str(prefix_i.representative)[1:]
        u_i = prefix_tree.topdown()
        # Check that we can descend the prefix tree to the correct context
        if not u_i.goDown(u[::-1]):
            raise ValueError('Could not descend context')
        if count:
            context_counts[u_i.value.id][x.ordValue] += count
    return context_counts


class CactoModel(object):
    """A non-parametric sequence model.
    """

    def __init__(self, seqs):
        self.prefix_tree = make_prefix_index(seqs)
        self.t = numpy.zeros((2 * len(self.prefix_tree), Value.valueSize), dtype=int)
        self.s = numpy.zeros((2 * len(self.prefix_tree), Value.valueSize), dtype=int)
        self._initialise()


    def _locate_context(self, u, topdownhistory=False):
        "Iterate down to the context u."
        if topdownhistory:
            i = self.prefix_tree.topdownhistory()
        else:
            i = self.prefix_tree.topdown()
        i.goDown(u[::-1])
        return i


    def log_context_counts(self, parent, it):
        """Visitor function to be used in callback descender
        to log context counts."""
        _logger.debug('Context counts: %-10s: %s',
            quote(prefixfor(it)), self.s[it.value.id])


    def log_table_counts(self, parent, it):
        """Visitor function to be used in callback descender
        to log table counts."""
        _logger.debug('Table counts: %-10s: %s',
                      quote(prefixfor(it)), self.t[it.value.id])


    def _initialise(self):
        """Initialise the table counts."""
        prefixes = dict()
        count_prefixes(self.prefix_tree, prefixes, self.prefix_tree.topdown())
        s = count_contexts(self.prefix_tree, prefixes)
        def initialise_vertex(parent, it):
            "Initialise the vertex the iterator points to."
            id_ = it.value.id
            for xord, count in enumerate(s[id_]):
                for _ in xrange(count):
                    self._initialise_with(xord, copy(it))
                    self.s[id_,xord] += 1
        seqan.CallbackDescender(initialise_vertex)(self.prefix_tree, history=True)
        #seqan.CallbackDescender(self.log_context_counts)(self.prefix_tree)
        assert (self.s == s).all()


    def _initialise_with(self, xord, i):
        """Take account of drawing x from the context at i in the prefix tree during
        model initialisation.
        """
        ulen = i.repLength
        du = self.d(ulen)
        tu = self._tu(i)
        oddsoldtable = (self.s[i.value.id,xord] + du * tu[xord]) / (
            self._p2(xord, i) * (self.theta(ulen) + du * tu.sum())
        )
        poldtable = oddsoldtable / (1 + oddsoldtable)
        if numpy.random.uniform() >= poldtable:
            # new table
            tu[xord] += 1
            # go up to the parent context if there is one
            sigmai = copy(i)
            if sigmai.goUp():
                self._initialise_with(xord, sigmai, s)


    def _tu(self, i):
        "The table counts for the given context."
        return self.t[i.value.id]


    def _su(self, i):
        "The prefix counts for the given context."
        return self.s[i.value.id]


    def theta(self, context_len):
        "Theta for the context length."
        return 1.


    def d(self, context_len):
        "Discount parameter for the context length."
        return 0.


    def _tu_children(self, i):
        "Get the counts of tables in the children."
        result = numpy.zeros(Value.valueSize, dtype=int)
        child = copy(i)
        if child.goDown():
            while True:
                result += self._tu(child)
                if not child.goRight():
                    break
        return result


    def _p2(self, xord, i):
        """Recursive function to determine likelihoods.

        - xord: The ordinal value of the draw.
        - i: A top down history iterator for the node in the
          prefix tree that represents u
        """
        ulen = i.repLength
        su = self._su(i)
        tu = self._tu(i)
        tu_children = self._tu_children(i)
        du = self.d(ulen)
        thetau = self.theta(ulen)
        # p(x|sigma(u))
        if i.goUp():
            p_x_sigmau = self._p2(xord, i)
        else:
            p_x_sigmau = uniformovervalues
        # Contribution from this node
        return (
            su[xord] + tu_children[xord] - du * tu[xord]
            + (thetau + du * tu.sum()) * p_x_sigmau
        ) / (
            thetau + su.sum() + tu_children.sum()
        )


    def _p(self, i, x, u, p_parent):
        """Recursive function used to determine likelihoods"""
        context_len = len(u)
        su = self._su(i)
        tu = self._tu(i)
        tu_children = self._tu_children(i)
        d = self.d(context_len)
        theta = self.theta(context_len)
        pG_x_given_u = (
            su[x]
            + tu_children[x]
            - d * tu[x]
            + (theta + d * tu.sum()) * p_parent
        ) / (
            theta + sum(su) + tu_children.sum()
        )
        _logger.info(
            '          : p_G(x=%s|u=%-15s) = %.3e',
            Value.fromOrdinal(x),
            quote(str(i.representative)[::-1]),
            pG_x_given_u)
        # We should keep descending if we matched the whole of the
        # representative so far and there is more tree to descend
        # that matches at least part of the rest of u
        #from IPython.core.debugger import Tracer
        #Tracer()()
        if (
            i.repLength < len(u)
            and i.representative == u[:i.repLength]
            and i.goDown(u[-1-i.repLength])
        ):
            return self._p(i, x, u, pG_x_given_u)
        else:
            # can't go any further down this context
            return pG_x_given_u


    def _predictive(self, xord, u):
        "p(x|u) where u is the context and x is the next symbol"
        return self._p(
            self.prefix_tree.topdown(),
            xord,
            u,
            uniformovervalues)


    def predictive(self, x, u):
        "p(x|u) where u is the context and x is the next symbol"
        _logger.info('Evaluating: p_G(x=%s|u=%-15s)', x, quote(u))
        return self._predictive(x.ordValue, u)
        #_logger.info('          : p_G(x=%s|u=%-15s) = %.3e', x, quote(u), p_x_given_u)
