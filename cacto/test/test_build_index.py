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
import sys
from . import fasta_file
from itertools import groupby, imap, chain
import cacto


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


def _count_contexts_descend(index, context_counts, i, loglevel=0):
    """Helper function for count_contexts().
    Descend the index counting how many children each node has.
    """
    # We need to calculate how many child occurrences we have
    # as are only interested in leaf occurrences of this node
    child_occurrences = 0
    representative = i.representative
    if i.goDown():
        while True:
            child_occurrences += i.countOccurrences
            count_contexts(index, context_counts, copy(i))
            if not i.goRight():
                break
    logger.log(
        loglevel,
        '%10s has %2d child occurrences',
        quote(representative),
        child_occurrences)
    return child_occurrences


def find_context(index, i):
    """Separate the representative of the iterator i into
    its first base and the remaining suffix.
    Return the base and an iterator locating the remaining suffix.

    This is essentially a suffix link perhaps modulo reversing of the suffixes.
    """
    # Separate representative into context and subsequent base given the context
    base = i.representative[0]
    context = i.representative[1:]
    context_i = index.TopDownIterator(index)
    #logger.debug('Finding context "%s" for base %s', context, base)
    if context_i.goDown(context):
        return base, context_i


def count_contexts(index, context_counts, i):
    """Count how many symbols follow each context.
    """
    # Descend the rest of the tree and return how many occurrences are in it
    child_occurrences = _count_contexts_descend(index, context_counts, copy(i))

    # we only have contexts for non-root nodes
    if i.representative:
        found_context = find_context(index, i)
        assert found_context
        base, context_i = found_context
        #logger.debug('Context %5s; base %s',
            #quote(context_i.representative), base)

        # Update counts, index by context's id then by the base value that follows
        #logger.debug('Context %5s has %2d occurrences and %2d child occurrences',
                    #quote(context_i.representative),
                    #context_i.countOccurrences,
                    #child_occurrences)
        counts = context_i.countOccurrences - child_occurrences
        if counts > 0:
            context_counts[context_i.value,base.ordValue] = counts
#         if len(i.representative) < 3:
#             logger.info('%-2s : %5d', i.representative, i.countOccurrences)



def quote(s):
    """Wrap the string in quotes."""
    return '"%s"' % s


def show_contexts(index, context_counts, i):
    """Show how many times each context occurs.
    """
    # We need to calculate how many child occurrences we have
    # as are only interested in leaf occurrences of this node
    for (vertex, base), count in context_counts.iteritems():
        i = index.TopDownIterator(index, vertex)
        logger.info(
            'Context %10s is followed by %s %3d times',
            quote(str(i.representative)[::-1]), 'ACGT'[base], count)


def _test_count_contexts():
    """Read the reversed sequences, split them into known sections and put them in a StringDNASet.
    """
    logger.info(sys._getframe().f_code.co_name)

    seqs = (
        'AACGGT',
        'AACGGA',
    )
    index = cacto.make_prefix_index(seqs)
    context_counts = dict()
    count_contexts(index, context_counts, index.TopDownIterator(index))
    show_contexts(index, context_counts, index.TopDownIterator(index))
    1/0



def build_desired_prefix_counts(seqs):
    from collections import defaultdict
    desired = defaultdict(int)
    for seq in seqs:
        print 'Seq:', seq
        for i in xrange(1, len(seq)+1):
            print 'Prefix:', seq[:i]
            desired[seq[:i]] += 1
    return desired


def test_count_prefixes():
    """Read the reversed sequences, split them into known sections
    and put them in a StringDNASet.
    """
    logger.info(sys._getframe().f_code.co_name)

    for seqs in (
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
    ):
        index = cacto.make_prefix_index(seqs)
        prefix_counts = dict()
        cacto.count_prefixes(index, prefix_counts, index.topdown())
        desired_results = build_desired_prefix_counts(seqs)
        print desired_results
        for i, count in prefix_counts.iteritems():
            prefix = str(i.representative)[::-1]
            logger.info(
                '%-10s is a prefix %2d times',
                quote(prefix), count)
            assert desired_results[prefix] == count
            del desired_results[prefix]
        assert not desired_results


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
    # No matter what the context we should see p = 1/4
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
    seqs = (
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
    #
    # No matter what the context we should see p = 1/4
    #
    for u in (
        'ATATATATATA',
        'A',
        'GC',
    ):
        x = seqan.DNA('T')
        model.predictive(x, u)
        #assert abs(.25 - model.predictive(x, u)) < 1e-15
    1/0

