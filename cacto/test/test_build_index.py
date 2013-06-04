#
# Copyright John Reid 2013
#

"""
Test building an index.
"""

import seqan, logging
from . import fasta_file
from copy import copy
from itertools import groupby, imap, chain

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def is_known(x):
    return 'N' != x


def split_sequence(seq):
    """Split a sequence into those sections that are known bases."""
    logger.info('Splitting: %s', seq)
    for k, g in groupby(seq, is_known):
        if k:
            yield ''.join(imap(str, g))

    
def test_split_sequence():
    result = list(split_sequence(seqan.StringDna5('NNACNGANGGN')))
    assert result[0] == 'AC', result[0]
    assert result[1] == 'GA', result[1]
    assert result[2] == 'GG', result[2]
    assert 3 == len(result)
   
    result = list(split_sequence('ACGTNNNNAAGG'))
    assert result[0] == 'ACGT', result[0]
    assert result[1] == 'AAGG', result[1]
    assert 2 == len(result)
    

def count_contexts(property_map, i):
    """Count how many symbols follow each context.
    """
    if len(i.representative) < 3:
        logging.info('%-2s : %5d', i.representative, i.countOccurrences)
    if i.goDown():
        while True:
            count_contexts(property_map, copy(i)) 
            if not i.goRight():
                break


def read_sequences(fasta):
    """Read the reversed sequences, split them into known sections and put them in a StringDnaSet.
    """
    num_bases, seqs_dna5, _ids = seqan.readFastaDna5(fasta_file(fasta), reverse=True)
    for _id, seq in zip(_ids, seqs_dna5):
        logger.info('%s: %d bases', _id, len(seq))
    logger.info('Read %d bases', num_bases)
    seqs_dna4 = seqan.StringDnaSet()
    for seq in chain.from_iterable(imap(split_sequence, seqs_dna5)):
        logger.info(seq)
        seqs_dna4.appendValue(seqan.StringDna(seq))
    index = seqan.IndexEsaDna(seqs_dna4)
    property_map = [None] * (2 * len(index))
    count_contexts(property_map, index.TopDownIterator(index))
   

def test_build_index():
    logger.info('Reading sequences')
    read_sequences(fasta_file('dm01r.fasta'))
