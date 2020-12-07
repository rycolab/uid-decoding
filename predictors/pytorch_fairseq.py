import logging
import os

import utils
from predictors.core import Predictor

from fairseq import checkpoint_utils, options, tasks
from fairseq import utils as fairseq_utils
from fairseq.sequence_generator import EnsembleModel
import torch
import numpy as np
import copy


def _initialize_fairseq(user_dir):
    logging.info("Setting up fairseq library...")
    if user_dir:
        args = type("", (), {"user_dir": user_dir})()
        fairseq_utils.import_user_module(args)

def get_fairseq_args(model_path, lang_pair):
    parser = options.get_generation_parser()
    input_args = ["--path", model_path, os.path.dirname(model_path)]
    if lang_pair:
        src, trg = lang_pair.split("-")
        input_args.extend(["--source-lang", src, "--target-lang", trg])
    return options.parse_args_and_arch(parser, input_args)


class FairseqPredictor(Predictor):
    """Predictor for using fairseq models."""
    name = 'fairseq'
    def __init__(self, args):
        super(FairseqPredictor, self).__init__()
        _initialize_fairseq(args.fairseq_user_dir)

        self.use_cuda = torch.cuda.is_available() and args.n_cpu_threads < 0
        fairseq_args = get_fairseq_args(args.fairseq_path, args.fairseq_lang_pair)

        # Setup task, e.g., translation
        task = tasks.setup_task(fairseq_args)
        source_dict = task.source_dictionary
        target_dict = task.target_dictionary
        self.src_vocab_size = len(source_dict) + 1
        self.trg_vocab_size = len(target_dict) + 1
        self.pad_id = target_dict.pad()
         # Load ensemble
        self.models = self.load_models(args.fairseq_path, task)
        self.model = EnsembleModel(self.models)
        self.model.eval()
        self.incremental_states = [{}]*len(self.models)


    def load_models(self, model_path, task):
        logging.info('Loading fairseq model(s) from {}'.format(model_path))
        models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'),
            task=task,
        )

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1,
                need_attn=False,
            )
            if self.use_cuda:
                model.cuda()
        return models

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
        
    @torch.no_grad()  
    def predict_next(self):
        """Call the fairseq model."""
        inputs = torch.LongTensor([self.consumed])
        
        if self.use_cuda:
            inputs = inputs.cuda()
        lprobs, _  = self.model.forward_decoder(
            inputs, self.encoder_outs, self.incremental_states)
        lprobs[:, self.pad_id] = utils.NEG_INF
        return np.array(lprobs[0].cpu() if self.use_cuda else lprobs[0], dtype=np.float64)
    
    @torch.no_grad()   
    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""

        src_tokens = torch.LongTensor([
            utils.oov_to_unk(src_sentence + [utils.EOS_ID],
                             self.src_vocab_size)])
        src_lengths = torch.LongTensor([len(src_sentence) + 1])
        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        self.encoder_outs = self.model.forward_encoder({
            'src_tokens': src_tokens,
            'src_lengths': src_lengths})

        self.consumed = [utils.GO_ID or utils.EOS_ID]
        self.reset_states()

    def reset_states(self, states=None):
         # Reset incremental states
        for i in range(len(self.models)):
            self.incremental_states[i] = {}
   
    def consume(self, word, i=None):
        """Append ``word`` to the current history."""
        self.consumed.append(word) if i is None else self.consumed[i].append(word)
    
    def get_empty_str_prob(self):
        return self.get_initial_dist()[utils.EOS_ID].item()

    @torch.no_grad()   
    def get_initial_dist(self):
        inputs = torch.LongTensor([[utils.GO_ID or utils.EOS_ID]])
        if self.use_cuda:
            inputs = inputs.cuda()
        
        lprobs, _ = self.model.forward_decoder(
            inputs, self.encoder_outs, [{}]*len(self.models)
        )
        return np.array(lprobs[0].cpu() if self.use_cuda else lprobs[0], dtype=np.float64)

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, self.incremental_states
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed, self.incremental_states = state

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]

    @staticmethod
    def add_args(parser):
        parser.add_argument("--fairseq_path", default="",
                       help="Points to the model file (*.pt) for the fairseq "
                       "predictor. Like --path in fairseq-interactive.")
        parser.add_argument("--fairseq_user_dir", default="",
                           help="fairseq user directory for additional models.")
        parser.add_argument("--fairseq_lang_pair", default="",
                           help="Language pair such as 'en-fr' for fairseq. Used "
                           "to load fairseq dictionaries")

    
