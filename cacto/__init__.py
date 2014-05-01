#
# Copyright John Reid 2013
#

"""
A python package for non-parametric sequence models.
"""


import logging
logger = logging.getLogger(__name__)

import numpy
import seqan
from copy import copy
from collections import defaultdict


ALPHABET_LEN = 4


def make_prefix_index(seqs):
    "Make an index out of the reverse of the sequences."
    sequences = seqan.StringDNASet()
    for seq in seqs:
        logger.info('Building prefix index from: %s', seq)
        sequences.appendValue(seqan.StringDNA(seq[::-1]))
    return seqan.IndexStringDNASetESA(sequences)


def count_prefixes(prefix_tree, prefix_counts=None, i=None):
    """Count how many times each prefix occurs in the prefix_tree.
    """
    from itertools import imap
    if prefix_counts is None:
        prefix_counts = dict()
    if i is None:
        i = prefix_tree.topdown()
    # Check all occurrences match
    assert [i.representative] * i.countOccurrences == \
        [prefix_tree.text[occ.i1][occ.i2:occ.i2+i.repLength]
            for occ in i.occurrences]
    prefix_count = sum(imap(
        lambda occ: occ.i2 + i.repLength == len(prefix_tree.text[occ.i1]),
        i.occurrences))
    occ = i.occurrences[0]
    logger.debug('Have %3d prefixes for: "%s"', prefix_count, str(i.representative)[::-1])
    if prefix_count:
        prefix_counts[copy(i)] = prefix_count

    if i.goDown():
        while True:
            count_prefixes(prefix_tree, prefix_counts, copy(i))
            if not i.goRight():
                break

    return prefix_counts


def count_contexts(prefix_tree, prefix_counts, alphabet_len=ALPHABET_LEN):
    """Create a dictionary mapping vertexes to counts. Each set of
    counts for a vertex reflects the number of times those bases follow
    the context that the vertex represents.
    """
    #context_counts = defaultdict(lambda: numpy.zeros(alphabet_len))
    context_counts = numpy.zeros((2 * len(prefix_tree), alphabet_len))
    for prefix_i, count in prefix_counts.iteritems():
        # context is all but last symbol, reversed
        prefix = str(prefix_i.representative)[::-1]
        u = prefix[:-1]
        #x = prefix_i.representative[prefix_i.repLength-1]
        x = prefix_i.representative[0]
        #logger.debug('prefix = "%s"', prefix)
        #logger.debug('u      =  %s', u)
        #logger.debug('x      =  %s%s', ' ' * len(u), x)
        assert prefix == u + str(x)
        #logger.debug(u[::-1])
        #logger.debug(str(prefix_i.representative)[1:])
        assert u[::-1] == str(prefix_i.representative)[1:]
        u_i = prefix_tree.topdown()
        if not u_i.goDown(u[::-1]):
            raise ValueError('Could not descend context')
        context_counts[u_i.value.id][x.ordValue] += count
    return context_counts


class CactoModel(object):
    """A non-parametric sequence model.
    """

    def __init__(self, seqs):
        self.prefix_tree = make_prefix_index(seqs)
        self.t = dict()
        self._empty_counts = numpy.zeros(ALPHABET_LEN)
        prefixes = dict()
        count_prefixes(self.prefix_tree, prefixes, self.prefix_tree.topdown())
        self.s = count_contexts(self.prefix_tree, prefixes)


    def _locate_context(self, u):
        "Iterate down to the context u."
        i = self.prefix_tree.topdown()
        i.goDown(u[::-1])
        return i


    def _tu(self, i):
        "The table counts for the given context."
        return self.t.get(i.value.id, self._empty_counts)


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
        result = numpy.zeros(ALPHABET_LEN)
        child = copy(i)
        if child.goDown():
            while True:
                result += self._tu(child)
                if not child.goRight():
                    break
        return result


    def _p(self, i, x, u, p_parent):
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
        logger.info(
            'p_G(%s|%s) = %.2e',
            i.representative.Value.fromOrdinal(x),
            str(i.representative)[::-1],
            pG_x_given_u)
        # We should keep descending if we matched the whole of the
        # representative so far and there is more tree to descend
        from IPython.core.debugger import Tracer
        #Tracer()()
        if (
            i.repLength < len(u)
            and i.representative == u[:i.repLength]
            and i.goDown(u[i.repLength:])
        ):
            return self._p(i, x, u, pG_x_given_u)
        else:
            # can't go any further down this context
            return pG_x_given_u


    def predictive(self, x, u):
        "p(x|u) where u is the context and x is the next symbol"
        logger.info('Evaluating: p_G(%s|%s)', x, u)
        p_x_given_u = self._p(
            self.prefix_tree.topdown(),
            x.ordValue,
            u,
            1./ALPHABET_LEN)
        logger.info('p(%s|%s) = %.3e', x, u, p_x_given_u)
        return p_x_given_u
