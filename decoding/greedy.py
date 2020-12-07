import utils
from decoding.core import Decoder, PartialHypothesis

class GreedyDecoder(Decoder):
   
    name = "greedy"
    def __init__(self, decoder_args):
        super(GreedyDecoder, self).__init__(decoder_args)
    
    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        hypothesis = PartialHypothesis(self.get_predictor_states(), self.calculate_stats)
        while hypothesis.get_last_word() != utils.EOS_ID and len(hypothesis) < self.max_len:
            ids, posterior, original_posterior = self.apply_predictor(
                                                    hypothesis if self.gumbel else None, 1)
            trgt_word = ids[0]
            if self.gumbel:
                hypothesis.base_score += original_posterior[0]
                hypothesis.score_breakdown.append(original_posterior[0])
            else: 
                hypothesis.score += posterior[0]
                hypothesis.score_breakdown.append(posterior[0])
            hypothesis.trgt_sentence.append(trgt_word)
            
            self.consume(trgt_word)
        self.add_full_hypo(hypothesis.generate_full_hypothesis())
        return self.full_hypos
