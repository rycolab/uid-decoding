import logging
import time
import utils

from collections import defaultdict
from datastructures.min_max_queue import MinMaxHeap
from datastructures.pointer_queue import PointerQueue
from decoding.core import Decoder, PartialHypothesis


class DijkstraTSDecoder(Decoder):
    
    name = "dijkstra_ts"
    def __init__(self, decoder_args):
        super(DijkstraTSDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.beam = decoder_args.beam if not self.gumbel else self.nbest
        self.stop_criterion = self._best_eos if decoder_args.early_stopping else self._all_eos

        self.size_threshold = self.beam*decoder_args.memory_threshold_coef\
            if decoder_args.memory_threshold_coef > 0 else utils.INF
        
    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        self.initialize_order_ds() 
        self.total_queue_size = 0
        
        while self.queue_order:
            _, t = next(self.queue_order)
            cur_queue = self.queues[t]
            score, hypo = cur_queue.popmax() 
            self.total_queue_size -= 1
            self.time_sync[t] -= 1

            if hypo.get_last_word() == utils.EOS_ID:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if self.stop_criterion(): # if stopping criterion are met
                    break
                self.update(cur_queue, t, forward_prune=True)
                continue

            if t == self.max_len:
                self.update(cur_queue, t)
                continue

            next_queue = self.queues[t+1]            
            for next_hypo in self._expand_hypo(hypo, self.beam):
                self.add_hypo(next_hypo, next_queue, t+1)
                
            self.update(cur_queue, t)
            self.update(next_queue, t+1)
        
        return self.get_full_hypos_sorted()  

    def initialize_order_ds(self):
        self.queues = [MinMaxHeap() for k in range(self.max_len+1)]
        self.queue_order = PointerQueue([0.0], reserve=self.max_len)
        self.time_sync = defaultdict(lambda: self.beam if self.beam > 0 else utils.INF)

        # Initialize BOS hypothesis
        self.queues[0].insert((0.0, PartialHypothesis(self.get_predictor_states(), self.calculate_stats)))
        self.time_sync[0] = 1
        
    def update(self, queue, t, forward_prune=False):
        # remove current best value associated with time step
        self.queue_order.popindex(t, default=None)

        #if beam used up at current time step, can prune hypotheses from older time steps
        if self.time_sync[t] <= 0:
            self.prune(t)

        #replace with next best value if anything left in queue
        if len(queue) > 0:
            self.queue_order[queue.peekmax()[0]] = t

        # if previous hypothesis was complete, reduce beam in next time steps
        if forward_prune:
            i = self.max_len
            while i > t:
                self.time_sync[i] -= 1
                if self.time_sync[i] <= 0:
                    self.prune(i)
                    return
                while len(self.queues[i]) > self.time_sync[i]:
                    # remove lowest scoring element since beam is getting "smaller"
                    self.queues[i].popmin()
                i-=1
    
    def prune(self, t):
        for i in range(t+1):
            self.queue_order.popindex(i, default=None)
            self.queues[i] = []
    
    def add_hypo(self, hypo, queue, t):
        score = self.get_adjusted_score(hypo)
        if len(queue) < self.time_sync[t]:
            queue.insert((score, hypo))
            if self.total_queue_size >= self.size_threshold:
                self.remove_one()
            else:
                self.total_queue_size += 1
        else:
            min_score = queue.peekmin()[0]
            if score > min_score:
                queue.replacemin((score, hypo))
        

    def remove_one(self):
        """ helper function for memory threshold"""
        for t, q in enumerate(self.queues):
            if len(q) > 0:
                q.popmax()
                if len(q) == 0:
                    self.queue_order.pop(self.score_by_t[t], default=None)
                return

    def _best_eos(self):
        """Returns true if the best hypothesis ends with </S>"""
        
        if len(self.full_hypos) < min(self.beam, self.nbest):
            return False
        if not self.not_monotonic:
            return True

        threshold = self.cur_best.total_score
        cur_scores = [self.get_max_pos_score(h[1]) for q in self.queues if q for h in q.a]
        return all([c < threshold for c in cur_scores])
            
    def _all_eos(self):
        """Returns true if the all hypotheses end with </S>"""
        if len(self.full_hypos) < self.beam:
            return False
        if not self.not_monotonic:
            return True

        threshold = sorted(self.full_hypos, reverse=True)[self.beam - 1].total_score
        cur_scores = [self.get_max_pos_score(h[1]) for q in self.queues if q for h in q.a]
        return all([c < threshold for c in cur_scores])

    @staticmethod
    def add_args(parser):
        parser.add_argument("--memory_threshold_coef", default=0, type=int,
                        help="total queue size will be set to `memory_threshold_coef`"
                         "* beam size. When capacity is exceeded, the worst scoring "
                         "hypothesis from the earliest time step will be discarded")
