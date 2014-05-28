#
# Copyright John Reid 2013, 2014
#

"""
A python package for non-parametric sequence models.
"""


import logging
_logger = logging.getLogger(__name__)

import numpy.random
import math
import seqan.descend
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


def curtail(string, length=40):
    """If the string is longer than length characters, shorten it and replace last
    characters with ellipses."""
    if len(string) > length:
        return '{0}...'.format(string[:length-3])
    else:
        return string


def make_prefix_index(seqs):
    "Make an index out of the reverse of the sequences."
    sequences = StringSet()
    for seq in seqs:
        _logger.debug('Building prefix index from: %s', curtail(seq))
        sequences.appendValue(String(seq[::-1]))
    return ESA(sequences)


def count_prefixes(prefixindex, prefix_counts=None, i=None):
    """Recursive function that counts how many times each
    prefix occurs in the prefixindex.

    Complexity: O(n log(n)) where n is the length of the text
    """
    from itertools import imap
    # Use a dictionary if no counts object provided
    if prefix_counts is None:
        prefix_counts = dict()
    # Use a root topdown iterator if none provided
    if i is None:
        i = prefixindex.topdown()
    # Double-check all occurrences match
    assert [i.representative] * i.numOccurrences == \
        [prefixindex.text[occ.i1][occ.i2:occ.i2+i.repLength]
            for occ in i.occurrences]
    # Count how many occurrences match the whole string
    prefix_count = i.numOccurrences
    copyi = i.copy()
    # Alternative calculation for prefix counts
    # alt_calculation = sum(imap(
        # lambda occ: occ.i2 + i.repLength == len(prefixindex.text[occ.i1]),
        # i.occurrences))
    # Recurse
    if i.goDown():
        while True:
            # Any occurrences in children do not match the whole string
            prefix_count -= i.numOccurrences
            # Recurse
            count_prefixes(prefixindex, prefix_counts, i.copy())
            if not i.goRight():
                break
    # Update counts if any occurrences matched the whole string
    if prefix_count:
        prefix_counts[copyi] = prefix_count
        _logger.debug('Have %3d prefixes of: "%s"',
                      prefix_count, str(i.representative)[::-1])
    # Check our prefix count against alternative calculation
    # assert prefix_count == alt_calculation
    # Return counts
    return prefix_counts


def count_contexts(prefixindex):
    """Count all the number of times each base is emitted in each context.

    Complexity: O(n log(n)) (I think)
    """
    context_counts = numpy.zeros((2 * len(prefixindex), Value.valueSize), dtype=int)
    def countcontextsforprefix(prefix_i, count):
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
        u_i = prefixindex.topdown()
        # Check that we can descend the prefix tree to the correct context
        if not u_i.goDown(u[::-1]):
            raise ValueError('Could not descend context')
        if count:
            context_counts[u_i.value.id][x.ordValue] += count
    seqan.findsuffixes(prefixindex.topdown(), countcontextsforprefix)
    return context_counts


def cactomodelfromseqs(seqs):
    """Build a Cacto model from a given set of sequences."""
    return CactoModel(make_prefix_index(seqs))


class CactoModel(object):
    """A non-parametric sequence model.
    """

    def __init__(self, prefixtree, theta=1., d=0.):
        if 0 > d or d >= 1:
            raise ValueError('d must satisfy 0 <= d < 1')
        if theta <= -d:
            raise ValueError('Theta must be > -d')
        self._theta = theta
        self._d = d
        self.prefixindex = prefixtree
        self.t = numpy.zeros((2 * len(self.prefixindex), Value.valueSize), dtype=int)
        self.s = numpy.zeros((2 * len(self.prefixindex), Value.valueSize), dtype=int)
        self._initialise()


    def _initialise(self):
        """Initialise the table counts."""
        s = count_contexts(self.prefixindex)
        def initialise_vertex(it):
            "Initialise the vertex the iterator points to."
            id_ = it.value.id
            for xord, count in enumerate(s[id_]):
                for _ in xrange(count):
                    self._initialise_with(xord, copy(it))
                    self.s[id_,xord] += 1
            return True
        descender = seqan.descend.Descender()
        descender.visitvertex = initialise_vertex
        descender.descend(self.prefixindex.topdownhistory())
        assert (self.s == s).all()


    def _initialise_with(self, xord, i):
        """Take account of drawing x from the context at i in the prefix tree during
        model initialisation.
        """
        ulen = i.repLength
        du = self.d(ulen)
        tu = self._tu(i)
        oddsoldtable = (self.s[i.value.id,xord] + du * tu[xord]) / (
            self.p_xord_given_ui(xord, i) * (self.theta(ulen) + du * tu.sum())
        )
        poldtable = oddsoldtable / (1 + oddsoldtable)
        if numpy.random.uniform() >= poldtable:
            # new table
            tu[xord] += 1
            # go up to the parent context if there is one
            if i.goUp():
                self._initialise_with(xord, i, s)


    def _locate_context(self, u, topdownhistory=False):
        "Iterate down to the context u."
        if topdownhistory:
            i = self.prefixindex.topdownhistory()
        else:
            i = self.prefixindex.topdown()
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


    def _tu(self, i):
        "The table counts for the given context."
        return self.t[i.value.id]


    def _tu_children(self, i):
        "Get the counts of tables in the children."
        result = numpy.zeros(Value.valueSize, dtype=int)
        if i.goDown():
            while True:
                result += self._tu(i)
                if not i.goRight():
                    break
            i.goUp()
        return result


    def _su(self, i):
        "The prefix counts for the given context."
        return self.s[i.value.id]


    def theta(self, context_len):
        "Theta for the context length."
        return self._theta


    def d(self, context_len):
        "Discount parameter for the context length."
        return self._d


    def calculateposterior(self):
        """Calculate the posterior p(x|u) for all emissions x and contexts u.
        Posterior is returned as a numpy array indexed by the vertex id of u
        then the ordinal of base x."""
        posterior = numpy.zeros((2 * len(self.prefixindex), Value.valueSize), dtype=float)
        def visitvertex(it):
            # What is the posterior for our parent node?
            if it.isRoot:
                # No: parent posterior is uniform distribution
                parent_posterior = numpy.ones(Value.valueSize) * uniformovervalues
            else:
                # Yes: parent posterior has already been calculated
                parent_posterior = posterior[it.nodeUp.id]
            ulen = it.repLength
            su = self._su(it)
            tu = self._tu(it)
            tu_children = self._tu_children(it)
            du = self.d(ulen)
            thetau = self.theta(ulen)
            # Contribution from this node
            posterior[it.value.id] = (
                su + tu_children - du * tu
                + (thetau + du * tu.sum()) * parent_posterior
            ) / (
                thetau + su.sum() + tu_children.sum()
            )
            # if it.representative == '': 1/0  # For debugging
            return True
        descender = seqan.descend.Descender()
        descender.visitvertex = visitvertex
        descender.descend(self.prefixindex.topdownhistory())
        return posterior


    def p_xord_given_ui(self, xord, i):
        """Recursive function to determine likelihood, p(x|u).

        - xord: The ordinal value of x.
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
            p_x_sigmau = self.p_xord_given_ui(xord, i)
        else:
            p_x_sigmau = uniformovervalues
        # Contribution from this node
        return (
            su[xord] + tu_children[xord] - du * tu[xord]
            + (thetau + du * tu.sum()) * p_x_sigmau
        ) / (
            thetau + su.sum() + tu_children.sum()
        )


    def _p_xord_given_u(self, i, x, u, p_parent):
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
        _logger.debug(
            '          : p_G(x=%s|u=%-15s) = %.3e',
            Value.fromOrdinal(x),
            quote(str(i.representative)[::-1]),
            pG_x_given_u)
        # We should keep descending if we matched the whole of the
        # representative so far and there is more tree to descend
        # that matches at least part of the rest of u
        if (
            i.repLength < len(u)
            and i.representative == u[:i.repLength]
            and i.goDown(u[-1-i.repLength])
        ):
            return self._p_xord_given_u(i, x, u, pG_x_given_u)
        else:
            # can't go any further down this context
            return pG_x_given_u


    def p_xord_given_u(self, xord, u):
        "p(x|u) where u is the context and xord is the ordinal value of the next symbol"
        return self._p_xord_given_u(
            self.prefixindex.topdownhistory(),
            xord,
            u,
            uniformovervalues)


    def p_x_given_u(self, x, u):
        """p(x|u) where u is the context and x is the next symbol. This is less efficient
        than"""
        _logger.debug('Evaluating: p_G(x=%s|u=%-15s)', x, quote(u))
        return self.p_xord_given_u(x.ordValue, u)
        #_logger.debug('          : p_G(x=%s|u=%-15s) = %.3e', x, quote(u), p_x_given_u)


    def seqsloglikelihood(self, seqs=None, seqsprefixindex=None, modelposterior=None):
        """The log likelihood of the sequences.

        The function builds a prefix tree of the sequences and counts how
        many emissions have been made for each context. Then the prefix
        tree is descended concurrently to the """
        if modelposterior is None:
            modelposterior = self.calculateposterior()
        if (seqs is None) == (seqsprefixindex is None):
            raise ValueError('Please specify exactly one of seqs or seqsprefixindex')
        if seqsprefixindex is None:
            seqsprefixindex = make_prefix_index(seqs)
        # The number of times each base is emitted in each context of the
        # sequences
        s = count_contexts(seqsprefixindex)
        class LLDescender(seqan.descend.ParallelDescender):
            "A descender that sums the log likelihood as it goes."
            def __init__(self_):
                self_.ll = 0.
            def visitvertex(self_, modelit, seqsit, stillsynced):
                self_.ll += (
                    s[seqsit.value.id]
                    * numpy.log(modelposterior[modelit.value.id])).sum()
                return True
        descender = LLDescender()
        descender.descend(self.prefixindex.topdownhistory(), seqsprefixindex.topdownhistory())
        return descender.ll
