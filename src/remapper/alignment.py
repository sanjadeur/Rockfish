import mappy
from Bio import SeqIO
import re

from typing import Tuple, Set, Dict, Optional

from .util import BEDPos


def make_aligner(reference_file: str) -> mappy.Aligner:
    aligner = mappy.Aligner(reference_file, preset='map-ont')  # Load or build index
    if not aligner:
        raise Exception("ERROR: failed to load/build index")

    return aligner


def get_motif_positions(reference: SeqIO, motif: str, index: int,
                        bed_pos: Optional[BEDPos]) -> Dict[str, Tuple[Set[int], Set[int]]]:
    motif_positions = dict()

    for chromosome, record in reference.items():
        reference = str(record.seq)

        # Forward strand
        fwd_matches = re.finditer(motif, reference, re.I)
        fwd_pos = set(m.start() + index for m in fwd_matches)

        # Reverse strand
        rev_matches = re.finditer(motif, mappy.revcomp(reference), re.I)
        rev_pos = set(len(reference) - (m.start() + index) - 1 for m in rev_matches)

        if bed_pos is not None:
            try:
                fwd_bed, rev_bed = bed_pos[chromosome]
                fwd_pos &= fwd_bed
                rev_pos &= rev_bed
            except KeyError:
                pass

        motif_positions[chromosome] = fwd_pos, rev_pos

    return motif_positions
