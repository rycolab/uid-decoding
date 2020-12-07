import copy
import numpy as np
import time

import utils, sampling_utils
from bisect import bisect
from decoding.core import Decoder, PartialHypothesis


class SamplingDecoder(Decoder):
    name = "sampling"
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SamplingDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        assert not self.gumbel
        
    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        hypos = [PartialHypothesis(copy.deepcopy(self.get_predictor_states()), 
                        self.calculate_stats) for i in range(self.nbest)]

        t = 0
        base_seed = self.nbest*self.seed
        while hypos and t < self.max_len:
            next_hypos = []
            for sen_seed, hypo in enumerate(hypos):
                np.random.seed(seed=base_seed+sen_seed)
                if hypo.get_last_word() == utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis())
                else:
                    self._expand_hypo(hypo)
                    next_hypos.append(hypo)
            hypos = next_hypos
            t+=1

        for hypo in hypos:
            hypo.score = self.get_adjusted_score(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
                
        return self.get_full_hypos_sorted()

    
    def _expand_hypo(self, hypo):

        self.set_predictor_states(hypo.predictor_states)
        ids, posterior, _ = self.apply_predictor()
        ind = self._sample(posterior)
        next_word = ids[ind]

        hypo.predictor_states = self.get_predictor_states()
        hypo.score += posterior[ind]
        hypo.score_breakdown.append(posterior[ind])
        hypo.trgt_sentence += [next_word]
        if self.calculate_stats:
            hypo.statistics.push(posterior[ind], np.max(posterior))
        self.consume(next_word)

    def _sample(self, posterior):
        return sampling_utils.log_multinomial_sample(posterior)

    def is_deterministic(self):
        return False


class NucleusSamplingDecoder(SamplingDecoder):
    name = "nucleus_sampling"
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(NucleusSamplingDecoder, self).__init__(decoder_args)
        self.log_nucleus_threshold = np.log(decoder_args.nucleus_threshold)

    def _sample(self, posterior):
        return sampling_utils.log_nucleus_multinomial_sample(posterior, nucleus_p=self.log_nucleus_threshold)[0]


    @staticmethod
    def add_args(parser):
        parser.add_argument('--nucleus_threshold', default=0.95, type=float, metavar='N',
                       help='implementation of Holtzman et. al 2019 p-nucleus sampling. '
                       "Value specifies probability core from which to consider "
                       "top items for sampling. Only compatible with 'sampling' "
                       "decoder.")
        
    
