import math

from tslearn import metrics

from genex.classes.Sequence import Sequence


def lb_keogh_sequence(seq_matching: Sequence, seq_enveloped: Sequence) -> float:
    """
    calculate lb keogh lower bound between query and sequence with envelope around query
    :param seq_matching:
    :param seq_enveloped:
    :return: lb keogh lower bound distance between query and sequence
    """
    try:
        assert len(seq_matching) == len(seq_enveloped)
    except AssertionError as ae:
        raise Exception('cluster.lb_keogh_sequence: two sequences must be of equal length to calculate lb_keogh')
    envelope_down, envelope_up = metrics.lb_envelope(seq_enveloped, radius=1)
    lb_k_sim = metrics.lb_keogh(seq_matching,
                                envelope_candidate=(envelope_down, envelope_up))
    return lb_k_sim / len(seq_matching)  # normalize


def lb_kim_sequence(candidate_seq, query_sequence):
    """
    Calculate lb kim lower bound between candidate and query sequence
    :param candidate_seq:
    :param query_sequence:
    :return: lb kim lower bound distance between query and sequence
    """

    lb_kim_sim = math.sqrt((candidate_seq[0] - query_sequence[0])**2 + (candidate_seq[-1] - query_sequence[-1])**2)

    return lb_kim_sim / 2.0  # normalize