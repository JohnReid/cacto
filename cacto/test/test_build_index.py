#
# Copyright John Reid 2013
#

"""
Test building an index.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import seqan
import cacto
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


def build_desired_prefix_counts(seqs):
    from collections import defaultdict
    desired = defaultdict(int)
    for seq in seqs:
        logger.info('Seq: %s', seq)
        for i in xrange(1, len(seq)+1):
            logger.info('Prefix: %s', seq[:i])
            desired[seq[:i]] += 1
    return desired


def build_desired_context_counts(seqs):
    desired_prefix_counts = build_desired_prefix_counts(seqs)
    countinit = lambda: numpy.zeros(4)
    desired = defaultdict(countinit)
    for prefix, count in desired_prefix_counts.iteritems():
        desired[prefix[:-1]][seqan.DNA(prefix[-1]).ordinalValue] += count
    return desired


prefix_seq_sets = (
    (
        'AACGGT',
        'AACGGA',
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
    logger.info(sys._getframe().f_code.co_name)
    for seqs in prefix_seq_sets:
        index = cacto.make_prefix_index(seqs)
        prefix_counts = dict()
        cacto.count_prefixes(index, prefix_counts, index.topdown())
        desired_results = build_desired_prefix_counts(seqs)
        for prefix, count in desired_results.iteritems():
            logger.info('Desired: %-10s is a prefix %2d times', prefix, count)
        for i, count in prefix_counts.iteritems():
            prefix = str(i.representative)[::-1]
            logger.info(
                '%-10s is a prefix %2d times',
                quote(prefix), count)
            assert desired_results[prefix] == count
            del desired_results[prefix]
        assert not desired_results  # check is empty to show we found all the
        # prefixes


def descend_prefix_tree(it, context_counts, desired_counts):
    if it.goDown():
        while True:
            context = str(it.representative)[::-1]
            desired_counts[context] -= context_counts[it.value.id]
            descend_prefix_tree(copy(it), context_counts)
            if not it.goRight():
                break


def test_count_contexts():
    logger.info(sys._getframe().f_code.co_name)
    for seqs in prefix_seq_sets:
        prefix_tree = cacto.make_prefix_index(seqs)
        prefixes = dict()
        cacto.count_prefixes(index, prefix_counts, index.topdown())
        context_counts = dict()
        count_contexts(prefix_tree, prefixes)
        desired_counts = build_desired_context_counts(seqs)
        descend_prefix_tree(prefix_tree.topDown(), context_counts, desired_counts)


def _test_empty_model_predictions():
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


def test_model_predictions():
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
