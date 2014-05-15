#
# Copyright John Reid 2013, 2014
#

"""
Test building an index.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import seqan
import cacto
import numpy
from cacto.test import fasta_file

import sys
from copy import copy
from itertools import groupby, imap, chain
from collections import defaultdict


def is_known(x):
    return 'N' != x


def split_sequence(seq):
    """Split a sequence into those sections that are known bases."""
    logger.info('Splitting: %s', seq)
    for k, g in groupby(seq, is_known):
        if k:
            yield ''.join(imap(str, g))


def test_split_sequence():
    logger.info(sys._getframe().f_code.co_name)

    result = list(split_sequence(seqan.StringDNA5('NNACNGANGGN')))
    assert result[0] == 'AC', result[0]
    assert result[1] == 'GA', result[1]
    assert result[2] == 'GG', result[2]
    assert 3 == len(result)

    result = list(split_sequence('ACGTNNNNAAGG'))
    assert result[0] == 'ACGT', result[0]
    assert result[1] == 'AAGG', result[1]
    assert 2 == len(result)


def read_sequences(fasta):
    # Read and reverse the sequences
    num_bases, seqs_dna5, _ids = seqan.readFastaDNA5(fasta, reverse=True)
    for _id, seq in zip(_ids, seqs_dna5):
        logger.info('%s: %d bases', _id, len(seq))
    logger.info('Read %d bases in total', num_bases)

    # Split the sequences into their known portions
    seqs_dna4 = seqan.StringDNASet()
    for seq in chain.from_iterable(imap(split_sequence, seqs_dna5)):
        logger.info(seq)
        seqs_dna4.appendValue(seqan.StringDNA(seq))
    logger.info('Split %d sequences with %d possibly ambiguous bases into %d sections totalling %d unambiguous bases',
                len(seqs_dna5), num_bases, len(seqs_dna4), sum(imap(len, seqs_dna4)))


def test_read_sequences():
    logger.info(sys._getframe().f_code.co_name)
    read_sequences(fasta_file('dm01r.fasta'))


def quote(s):
    """Wrap the string in quotes."""
    return '"%s"' % s


def prefixfor(it):
    """The prefix for an iterator is the reverse of its representative."""
    return str(it.representative)[::-1]


def build_desired_prefix_counts(seqs):
    from collections import defaultdict
    desired = dict()
    for seq in seqs:
        #logger.debug('Seq: %s', seq)
        for i in xrange(1, len(seq)+1):
            prefix = seq[:i]
            #logger.debug('Prefix: %s', prefix)
            desired[prefix] = desired.get(prefix, 0) + 1
    return desired


def build_desired_context_counts(seqs):
    """Alternative algorithm to count symbols following each context."""
    desired_prefix_counts = build_desired_prefix_counts(seqs)
    countinit = lambda: numpy.zeros(4)
    desired = defaultdict(countinit)
    for prefix, count in desired_prefix_counts.iteritems():
        desired[prefix[:-1]][seqan.DNA(prefix[-1]).ordValue] += count
    return desired


prefix_seq_sets = (
    (
        'AACGGT',
        'AACGGA',
    ),
    (
        'GAACGGT',
        'CAACGGA',
    ),
    (
        'TAACGG',
        'AAACGG',
    ),
    (
        'TAACGG',
        'AAACGG',
    ),
    (
        'AAAA',
        'TTAA',
        'AAT',
    ),
    (
        'TCCTAAT',
        'GTTGCA',
        'AT',
    ),
    (
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'AA',
        'AC',
        'AG',
        'AT',
        'AA',
        'AC',
        'AG',
        'AT',
        'AA',
        'AC',
        'AG',
        'AT',
    ),
)


def test_count_prefixes():
    """Makes a prefix index and then counts the prefixes. Compares these
    counts against counts created by a different algorithm."""
    logger.info(sys._getframe().f_code.co_name)
    for seqs in prefix_seq_sets:
        index = cacto.make_prefix_index(seqs)
        prefix_counts = cacto.count_prefixes(index)
        desired_results = build_desired_prefix_counts(seqs)
        for prefix, count in desired_results.iteritems():
            logger.info('Desired: %-10s is a prefix %2d times', prefix, count)
        for i, count in prefix_counts.iteritems():
            prefix = prefixfor(i)
            logger.info(
                '%-10s is a prefix %2d times',
                quote(prefix), count)
            assert desired_results[prefix] == count
            del desired_results[prefix]
        # check is empty to show we found all the prefixes
        assert not desired_results


def remove_counts(it, context_counts, desired_counts):
    """Remove the counts in context_counts from desired_counts.
    We expect both to agree and there will be no counts left
    in desired_counts."""
    context = prefixfor(it)
    logger.debug('Removing counts for %s', quote(context))
    desired_counts[context] -= context_counts[it.value.id]
    if 0 == desired_counts[context].sum():
        del desired_counts[context]
    if it.goDown():
        while True:
            remove_counts(copy(it), context_counts, desired_counts)
            if not it.goRight():
                break


def test_count_contexts():
    """Count how many of each symbol follow each context."""
    logger.info(sys._getframe().f_code.co_name)
    for seqs in prefix_seq_sets:
        prefix_tree = cacto.make_prefix_index(seqs)
        #cacto.log_prefix_tree(prefix_tree.topdown())
        def logprefixcounts(parent, it):
            logger.debug('Prefix tree: "%s"', str(it.representative)[::-1])
        #seqan.CallbackDescender(logprefixcounts)(prefix_tree)
        prefix_counts = cacto.count_prefixes(prefix_tree)
        context_counts = cacto.count_contexts(prefix_tree, prefix_counts)
        def logcontextcounts(parent, it):
            logger.debug('Context counts: %-10s: %s',
                quote(prefixfor(it)), context_counts[it.value.id])
        seqan.CallbackDescender(logcontextcounts)(prefix_tree)
        desired_counts = build_desired_context_counts(seqs)
        for context, counts in desired_counts.iteritems():
            logger.debug('Desired counts: %-10s: %s', quote(context), counts)
        remove_counts(prefix_tree.topdown(), context_counts, desired_counts)
        if desired_counts:
            for context, counts in desired_counts.iteritems():
                logger.error('Desired counts remaining: %-10s: %s',
                             quote(context), counts)
            raise ValueError('Desired counts did not match calculated counts')


def _test_empty_model_predictions():
    """Currently dumps core due to seqan bug."""
    seqs = tuple('',)
    model = cacto.CactoModel(seqs)
    #
    # No matter what the context we should see p = 1/4
    #
    for u in (
        '',
        'A',
        'GC',
    ):
        x = seqan.DNA('A')
        logger.info('p(%s|%s) = %.3e', x, u, model.predictive(x, u))
        assert abs(.25 - model.p(x, u)) < 1e-15


def test_simple_model_predictions():
    seqs = (
        'A',
        'C',
        'G',
        'T',
    )
    model = cacto.CactoModel(seqs)
    #
    # No matter what the context we should see p(x|u) = 1/4
    #
    for u in (
        '',
        'A',
        'GC',
    ):
        x = seqan.DNA('A')
        logger.info('p(%s|%s) = %.3e', x, u, model.predictive(x, u))
        assert abs(.25 - model.predictive(x, u)) < 1e-15


def _test_model_predictions():
    import seqan.io.graphtool
    seqs = (
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'ATATATATATAT',
        'AA',
        'AC',
        'AG',
        'AT',
        'AA',
        'AC',
        'AG',
        'AT',
        'AA',
        'AC',
        'AG',
        'AT',
    )
    model = cacto.CactoModel(seqs)
    for u in (
        'ATATATATATA',
        'A',
        'GC',
    ):
        x = seqan.DNA('T')
        model.predictive(x, u)
        #assert abs(.25 - model.predictive(x, u)) < 1e-15
    builder = seqan.io.graphtool.Builder(model.prefix_tree)
    seqan.io.graphtool.GT.graph_draw(
        builder.graph,
        pos=seqan.io.graphtool.GT.sfdp_layout(builder.graph),
        vertex_size=2,
        vertex_fill_color="lightgrey",
        vertex_font_size=8,
        vertex_text=builder.map_vertices(lambda it: '{0} {1} {2} {3}'.format(*map(int, model._su(it)))),
        vertex_pen_width=seqan.io.graphtool.root_vertex_property(builder),
        edge_text=seqan.io.graphtool.edge_labels_for_output(builder),
        edge_color=seqan.io.graphtool.color_edges_by_first_symbol(builder),
        edge_end_marker="none",
        edge_pen_width=2,
        #edge_dash_style=seqan.io.graphtool.dash_non_suffix_edges(builder, suffix),
        #edge_pen_width=builder.edge_lengths,
        #output="graphtool.png"
    )


if '__main__' == __name__:
    test_count_prefixes()
    test_model_predictions()
