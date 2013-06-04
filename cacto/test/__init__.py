#
# Copyright John Reid 2013
#

"""
Test code.
"""

import os, seqan

fasta_dir = os.path.join(os.path.dirname(__file__), 'fasta')

def fasta_file(filename):
    return os.path.join(fasta_dir, filename)

