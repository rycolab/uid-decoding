from abc import abstractmethod
import copy
import sys
import utils
import logging
import numpy as np
from runstats import Statistics



class Hypothesis:
    """Complete translation hypotheses are represented by an instance
    of this class. We store the produced sentence, the combined score,
    and a score breakdown to the separate predictor scores.
    """
    
    def __init__(self, trgt_sentence, total_score, score_breakdown = [], base_score=0., statistics=None):
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown
        self.base_score = base_score
        self.statistics = statistics

    def __repr__(self):
        """Returns a string representation of this hypothesis."""
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)
    def __len__(self):
        return len(self.trgt_sentence)

    def __lt__(self, other):
        return self.total_score < other.total_score
    

class PartialHypothesis(object):
    """Represents a partial hypothesis in various decoders. """
    
    def __init__(self, initial_states = None, use_stats=True):
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score, self.base_score = 0.0, 0.0
        self.score_breakdown = []
        self.word_to_consume = None
        self.statistics = Statistics() and self.statistics.push(0,0) if use_stats else None

    def __repr__(self):
        """Returns a string representation of this hypothesis."""
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.score)
    def __len__(self):
        return len(self.trgt_sentence)

    def __lt__(self, other):
        return self.score < other.score

    def __add__(self, other):
        return self.trgt_sentence + other
    
    def get_last_word(self):
        """Get the last word in the translation prefix. """
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]
        
    
    def generate_full_hypothesis(self):
        """Create a ``Hypothesis`` instance from this hypothesis. """
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown, self.base_score, self.statistics)
    
    def _new_partial_hypo(self, states, word, score, base_score=None, breakdown=None, cur_max=None):
        new_hypo = PartialHypothesis(states, use_stats=False)
        new_hypo.score = score 
        new_hypo.base_score = base_score 
        new_hypo.score_breakdown = copy.copy(self.score_breakdown)
        new_hypo.score_breakdown.append(breakdown if breakdown is not None else score)
        new_hypo.trgt_sentence = self.trgt_sentence + [word]
        if self.statistics is not None:
            new_hypo.statistics = self.statistics.copy()
            new_hypo.statistics.push(new_hypo.score_breakdown[-1], cur_max)
        
        return new_hypo

    def expand(self, word, new_states, score, score_breakdown):
        return self._new_partial_hypo(new_states, word, score, score_breakdown)
    
    def cheap_expand(self, word, score, base_score=None, breakdown=None, states=None, cur_max=None):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability. Does NOT update the
        predictor states but adds a flag which signals that the last 
        word in this hypothesis has not been consumed yet by the 
        predictors. This can save memory because we can reuse the 
        current state for many hypothesis. It also saves computation
        as we do not consume words which are then discarded anyway by
        the search procedure.
        
        """
        hypo = self._new_partial_hypo(states,
                                     int(word), score,
                                     base_score=base_score,
                                     breakdown=breakdown,
                                     cur_max=cur_max)
        hypo.word_to_consume = int(word)
        return hypo

    def get_score_variance(self, val=None):
        if val is not None:
            return self.statistics.pos_variance(val,ddof=0) 
        return self.statistics.variance(ddof=0)

    def get_score_max(self, val=None):
        if val is not None:
            return -self.statistics.pos_minimum(val) 
        return -self.statistics.minimum()

    def get_local_variance(self, val=None):
        if val is not None:
            return self.statistics.pos_local_variance(val,ddof=0) 
        return self.statistics.local_variance(ddof=0) 
        
    def get_score_greedy(self, val=None):
        if val is not None:
            val, cur_max = val
            return self.statistics.pos_max_offset(val, cur_max) 
        return self.statistics.max_offset()

    def get_squares(self, val=None):
        if val is not None:
            return self.statistics.pos_squares(val) 
        return self.statistics.squares() 

    
class Decoder(object):    
    """A ``Decoder`` instance represents a particular search strategy
    such as A*, beam search, greedy search etc. Decisions are made 
    based on the outputs of one or many predictors, which are 
    maintained by the ``Decoder`` instance.
    
    Decoders are observable. They fire notifications after 
    apply_predictors has been called. 
    """
    
    def __init__(self, decoder_args):
        """Initializes the decoder instance with no predictors.
        """
        super(Decoder, self).__init__()
        self.max_len_factor = decoder_args.max_len_factor
        self.predictor = None # Tuples (predictor, weight)
        self.predictor_names = []
        self.gumbel = decoder_args.gumbel
        self.seed = decoder_args.seed
        self.allow_unk_in_output = decoder_args.allow_unk_in_output
        self.nbest = 1 # length of n-best list
        self.combine_posteriors = self._combine_posteriors_simple
        self.current_sen_id = -1
        self.apply_predictor_count = 0
        self.temperature = decoder_args.temperature
        self.add_incomplete = decoder_args.add_incomplete
        self.calculate_stats = not decoder_args.no_statistics and not self.gumbel
        self.length_norm = decoder_args.length_normalization
        self.variance_reg = decoder_args.variance_regularizer
        self.local_variance_reg = decoder_args.local_variance_regularizer
        self.max_reg = decoder_args.max_regularizer
        self.greedy_reg = decoder_args.greedy_regularizer
        self.square_reg = decoder_args.square_regularizer
         # score function will be monotonic without modifications to scoring function
        self.not_monotonic = any([self.variance_reg,self.local_variance_reg, self.max_reg, 
                                self.greedy_reg, self.square_reg, self.length_norm]) 
        if any([self.variance_reg,self.local_variance_reg, self.max_reg,
            self.greedy_reg, self.square_reg]) and not self.calculate_stats:
            logging.fatal("Must use statistics with regularizers. Cannot use with Gumbel stochastic decoding")
            sys.exit(1)
        # if self.not_monotonic and decoder_args.early_stopping:
        #     logging.warn("Using early stopping with non-monotonic scoring function. Behavior "
        #         "may not be defined!")


    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass
        
    def is_deterministic(self):
        return not self.gumbel

    def get_inclusion_prob_estimate(self, src, hypo, kau=None, **kwargs):
        if self.gumbel:
            assert kau is not None
            zet = np.exp(hypo.base_score - kau)
            return hypo.base_score - kau - zet/2 + (zet**2)/24 - (zet**4)/2880 
        return hypo.total_score

    def add_predictor(self, name, predictor):
        """Adds a predictor to the decoder. This means that this 
        predictor is going to be used to predict the next target word
        (see ``predict_next``)
        
        Args:
            name (string): Predictor name like 'nmt' or 'fst'
            predictor (Predictor): Predictor instance
            weight (float): Predictor weight
        """
        self.predictor = predictor
    
    def remove_predictor(self):
        """Removes all predictors of this decoder. """
        self.predictor = None
    
    def has_predictor(self):
        """Returns true if predictors have been added to the decoder. """
        return self.predictor is not None
    
    def consume(self, word, i=None):
        """Calls ``consume()`` on all predictors. """
        self.predictor.consume(word)

    def get_initial_dist(self):
        return utils.log_softmax(self.predictor.get_initial_dist(), self.temperature)
    
    def _get_non_zero_words(self, predictor, posterior):
        """Get the set of words from the predictor posteriors which 
        have non-zero probability. This set of words is then passed
        through to the open vocabulary predictors.

        This method assumes that both arguments are not empty.

        Args:
            bounded_predictor: predictor
            bounded_posterior: Corresponding posterior.

        Returns:
            Iterable with all words with non-zero probability.
        """

        fin_probs = np.isfinite(posterior)
        return [i for i, b in enumerate(fin_probs) if b]
    
    def apply_predictor(self, hypo=None, top_n=0):
        """Get the distribution over the next word by combining the
        predictor scores.

        Args:
            top_n (int): If positive, return only the best n words.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        assert hypo is not None or not self.gumbel
        self.apply_predictor_count += 1
        # Get posteriors
        posterior = self.predictor.predict_next()
        posterior = utils.log_softmax(posterior, temperature=self.temperature)
        # numerical stability check
        assert len(posterior) - np.count_nonzero(posterior) <= 1
        
        non_zero_words = self._get_non_zero_words(self.predictor,
                                                  posterior)
        if len(non_zero_words) == 0: # Special case: no word is possible
            non_zero_words = set([utils.EOS_ID])

        if self.gumbel:
            gumbel_full_posterior = self.gumbelify(hypo, posterior)
            ids, posterior, original_posterior = self.combine_posteriors(
                non_zero_words, gumbel_full_posterior, self.predictor.get_unk_probability(posterior),
                top_n=top_n, original_posterior=posterior) 
        else:
            ids, posterior, original_posterior = self.combine_posteriors(
                non_zero_words, posterior, self.predictor.get_unk_probability(posterior), top_n=top_n) 
                
        assert self.allow_unk_in_output or not utils.UNK_ID in ids
        return ids, posterior, original_posterior


    def gumbelify(self, hypo, posterior):
        vf = np.vectorize(lambda x: self.get_pos_score(hypo, x) - self.get_adjusted_score(hypo))
        shifted_posterior = vf(posterior)
        shifted_posterior = utils.log_softmax(shifted_posterior)

        gumbels = np.random.gumbel(loc=0, scale=1, size=shifted_posterior.shape)
        gumbel_posterior = shifted_posterior + gumbels + hypo.base_score
        Z = np.max(gumbel_posterior)

        v = hypo.score - gumbel_posterior + utils.log1mexp_basic(gumbel_posterior - Z)
        gumbel_full_posterior = hypo.score - np.maximum(0, v) - utils.log1pexp_basic(-np.abs(v))

        # make sure invalid tokens still have neg inf log probability
        gumbel_full_posterior[(posterior == utils.NEG_INF).nonzero()] == utils.NEG_INF
        return gumbel_full_posterior

    
    def _expand_hypo(self, hypo, limit=0, return_dist=False):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """

        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        ids, posterior, original_posterior = self.apply_predictor(hypo, limit)

        max_score = np.max(posterior)
        new_states = self.get_predictor_states()
        new_hypos = [hypo.cheap_expand(
                        trgt_word,
                        posterior[idx] if self.gumbel else posterior[idx] + hypo.score,
                        base_score=original_posterior[idx] + hypo.base_score if self.gumbel else hypo.base_score,
                        breakdown=original_posterior[idx] if self.gumbel else posterior[idx],
                        states=new_states,
                        cur_max=max_score
                        ) for idx, trgt_word in enumerate(ids)]
        if return_dist:
            return new_hypos, posterior
        return new_hypos


    def get_adjusted_score(self, hypo):
        """Combines hypo score with penalties/rewards.""" 
        current_score = hypo.score
        if self.gumbel:
            return current_score
        if self.variance_reg:
            current_score -= self.variance_reg*hypo.get_score_variance()
        if self.max_reg:
            current_score -= self.max_reg*hypo.get_score_max()
        if self.local_variance_reg:
            current_score -= self.local_variance_reg*hypo.get_local_variance()
        if self.greedy_reg:
            current_score -= self.greedy_reg*hypo.get_score_greedy()
        if self.square_reg:
            current_score -= self.square_reg*hypo.get_squares()
        if self.length_norm: 
            current_score /= len(hypo)

        return current_score

    def get_pos_score(self, hypo, val, max_=None):
        """Combines hypo score with cost estimate from next round.""" 
        pos_score = hypo.score + val     
        if self.variance_reg:
            pos_score -= self.variance_reg*hypo.get_score_variance(val)
        if self.max_reg:
            pos_score -= self.max_reg*hypo.get_score_max(val)
        if self.local_variance_reg:
            pos_score -= self.local_variance_reg*hypo.get_local_variance(val)
        if self.greedy_reg:
            pos_score -= self.greedy_reg*hypo.get_score_greedy((val, max_))
        if self.square_reg:
            pos_score -= self.square_reg*hypo.get_squares(val)
        if self.length_norm: 
            pos_score /= len(hypo)

        return pos_score

    def get_max_pos_score(self, hypo):
        """For non monotonic regularizers.
        Returns maximum possible score given current values.""" 
        current_score = hypo.score
        if self.variance_reg:
            current_score -= self.variance_reg*hypo.get_score_variance()*len(hypo)/self.max_len
        if self.local_variance_reg:
            current_score -= self.local_variance_reg*hypo.get_local_variance()*len(hypo)/self.max_len
        if self.length_norm: 
            current_score /= self.max_len
        return current_score

    
    def _combine_posteriors_simple(self,
                                      non_zero_words,
                                      posterior,
                                      unk_prob,
                                      top_n=0,
                                      original_posterior=None):
        """        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if top_n > 0:
            non_zero_words = utils.argmax_n(posterior, top_n)

        scores_func = np.vectorize(lambda x: utils.common_get(posterior, x, unk_prob))
        scores = scores_func(non_zero_words)

        orig_scores = None
        if original_posterior is not None:
            scores_func = np.vectorize(lambda x: utils.common_get(original_posterior, x, unk_prob))
            orig_scores = scores_func(non_zero_words)

        return non_zero_words, scores, orig_scores

    
    def set_current_sen_id(self, sen_id):
        self.current_sen_id = sen_id - 1  # -1 because incremented in init()
            
    def initialize_predictor(self, src_sentence):
        """First, increases the sentence id counter and calls
        ``initialize()`` on all predictors.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        if not self.is_deterministic():
            np.random.seed(seed=self.seed)
        self.max_len = int(np.ceil(self.max_len_factor * len(src_sentence)))
        self.full_hypos = []
        self.current_sen_id += 1
        self.predictor.set_current_sen_id(self.current_sen_id)
        self.predictor.initialize(src_sentence)
    
    def add_full_hypo(self, hypo):
        """Adds a new full hypothesis to ``full_hypos``. This can be
        used by implementing subclasses to add a new hypothesis to the
        result set. 
        
        Args:
            hypo (Hypothesis): New complete hypothesis
        """
        if len(self.full_hypos) == 0 or hypo.total_score > self.cur_best.total_score:
            self.cur_best = hypo
        self.full_hypos.append(hypo)
    
    def get_full_hypos_sorted(self, additional_hypos=None):
        """Returns ``full_hypos`` sorted by the total score. Can be 
        used by implementing subclasses as return value of
        ``decode``
        
        Returns:
            list. ``full_hypos`` sorted by ``total_score``.
        """

        if additional_hypos is not None:
            incompletes = []
            for hypo in additional_hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 
                else: 
                    incompletes.append(hypo)

            if not self.full_hypos:
                logging.warn("No complete hypotheses found")

            if len(self.full_hypos) < self.nbest and self.add_incomplete:
                logging.warn("Adding incomplete hypotheses as candidates")
                incompletes.sort(key=lambda hypo: hypo.score, reverse=True)
                for hypo in incompletes[:self.nbest - len(self.full_hypos)]:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 

        return sorted(self.full_hypos,
                      key=lambda hypo: hypo.total_score,
                      reverse=True)

    def get_empty_hypo(self):
        hypo = PartialHypothesis(use_stats=True)
        score = self.predictor.get_empty_str_prob()

        hypo.score += score
        hypo.score_breakdown.append(score)
        hypo.trgt_sentence += [utils.EOS_ID]

        hypo.score = self.get_adjusted_score(hypo)
        return hypo

    
    def set_predictor_states(self, states):
        """Calls ``set_state()`` on all predictors. """
        self.predictor.set_state(states)
    
    def get_predictor_states(self):
        """Calls ``get_state()`` on all predictors. """
        return self.predictor.get_state()

    
    @abstractmethod
    def decode(self, src_sentence):
        """Decodes a single source sentence. This method has to be 
        implemented by subclasses. It contains the core of the 
        implemented search strategy ``src_sentence`` is a list of
        source word ids representing the source sentence without
        <S> or </S> symbols. This method returns a list of hypotheses,
        order descending by score such that the first entry is the best
        decoding result. Implementations should delegate the scoring of
        hypotheses to the predictors via ``apply_predictors()``, and
        organize predictor states with the methods ``consume()``,
        ``get_predictor_states()`` and ``set_predictor_states()``. In
        this way, the decoder is decoupled from the scoring modules.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        raise NotImplementedError
