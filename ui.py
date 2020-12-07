import argparse
import logging
import os
import sys
import platform
import decoding


def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


def run_diagnostics():
    """Check availability of external libraries."""
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    if sys.version_info > (3, 0):
        print("Checking Python3.... %sOK (%s)%s" 
              % (OKGREEN, platform.python_version(), ENDC))
    else:
        print("Checking Python3.... %sNOT FOUND %s%s"
              % (FAIL, sys.version_info, ENDC))
        print("Please upgrade to Python 3!")
    try:
        import torch
        print("Checking PyTorch.... %sOK (%s)%s"
              % (OKGREEN, torch.__version__, ENDC))
    except ImportError:
        print("Checking PyTorch.... %sNOT FOUND%s" % (FAIL, ENDC))
        print("PyTorch is not available. This affects the following "
              "components: Predictors: fairseq, onmtpy. Check the "
              "documentation for further instructions.")
    try:
        import fairseq
        print("Checking fairseq.... %sOK (%s)%s"
              % (OKGREEN, fairseq.__version__, ENDC))
    except ImportError:
        print("Checking fairseq.... %sNOT FOUND%s" % (FAIL, ENDC))
        print("fairseq is not available. This affects the following "
              "components: Predictors: fairseq. Check the "
              "documentation for further instructions.")


def get_parser():
    """Get the parser object which is used to build the configuration
    argument ``args``. This is a helper method for ``get_args()``
    TODO: Decentralize configuration
    
    Returns:
        ArgumentParser. The pre-filled parser object
    """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    ## General options
    group = parser.add_argument_group('General options')
    group.add_argument('--config_file', 
                        help="Configuration file in standard .ini format. NOTE:"
                        " Configuration file overrides command line arguments")
    group.add_argument("--run_diagnostics", default=False, action="store_true",
                       help="Run diagnostics and check availability of "
                       "external libraries.")
    group.add_argument("--verbosity", default="info",
                        choices=['debug', 'info', 'warn', 'error'],
                        help="Log level: debug,info,warn,error")
    group.add_argument("--range", default="",
                        help="Defines the range of sentences to be processed. "
                        "Syntax is equal to HiFSTs printstrings and lmerts "
                        "idxrange parameter: <start-idx>:<end-idx> (both "
                        "inclusive, start with 1). E.g. 2:5 means: skip the "
                        "first sentence, process next 4 sentences. If this "
                        "points to a file, we grap sentence IDs to translate "
                        "from that file and delete the decoded IDs. This can "
                        "be used for distributed decoding.")
    group.add_argument("--input_file", default="",
                        help="Path to source. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be appropriately formatted for "
                        "the specified preprocessing procedure, e.g. use word IDs "
                        "instead of their string representations if preprocessing "
                        "set to 'id'")
    group.add_argument("--trgt_file", default="",
                        help="Path to target test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be appropriately formatted for "
                        "the specified preprocessing procedure, e.g. use word IDs "
                        "instead of their string representations if preprocessing "
                        "set to 'id'")
    group.add_argument("--indexing_scheme", default="fairseq",
                        choices=['t2t', 'fairseq'],
                        help="This parameter defines the reserved IDs.\n\n"
                        "* 't2t': unk: 3, <s>: 2, </s>: 1.\n"
                        "* 'fairseq': unk: 3, <s>: 0, </s>: 2.")
    group.add_argument("--ignore_sanity_checks", default=False, type='bool',
                       help="SGNMT terminates when a sanity check fails by "
                       "default. Set this to true to ignore sanity checks.")
    group.add_argument("--input_method", default="file",
                        choices=['dummy', 'file', 'shell', 'stdin'],
                        help="This parameter controls how the input to SGNMT "
                        "is provided. SGNMT supports three modes:\n\n"
                        "* 'dummy': Use dummy source sentences.\n"
                        "* 'file': Read test sentences from a plain text file"
                            "specified by --src_test.\n"
                        "* 'shell': Start SGNMT in an interactive shell.\n"
                        "* 'stdin': Test sentences are read from stdin\n\n")
    group.add_argument("--log_sum",  default="log",
                        choices=['tropical', 'log'],
                        help="Controls how to compute the sum in the log "
                        "space, i.e. how to compute log(exp(l1)+exp(l2)) for "
                        "log values l1,l2.\n\n"
                        "* 'tropical': approximate with max(l1,l2)\n"
                        "* 'log': Use logsumexp in scipy")
    group.add_argument("--n_cpu_threads", default=-1, type=int,
                        help="Set the number of CPU threads.")
    
    ## Decoding options
    group = parser.add_argument_group('Decoding options')
    group.add_argument("--decoder", default=None, choices=decoding.DECODER_REGISTRY.keys(),
                        help="Strategy for traversing the search space which "
                        "is spanned by the predictors.\n")
    group.add_argument("--beam", default=0, type=int,
                        help="Size of beam. For 'dijkstra' it limits the capacity"
                        " of the queue. Use --beam 0 for unlimited capacity.")
    group.add_argument("--allow_unk_in_output", default=True, type='bool',
                        help="If false, remove all UNKs in the final "
                        "posteriors. Predictor distributions can still "
                        "produce UNKs, but they have to be replaced by "
                        "other words by other predictors")
    group.add_argument("--max_len_factor", default=2.0, type=float,
                        help="Limits the length of hypotheses to avoid "
                        "infinity loops in search strategies for unbounded "
                        "search spaces. The length of any translation is "
                        "limited to max_len_factor times the length of the "
                        "source sentence.")
    group.add_argument("--early_stopping", default=True, type='bool',
                        help="Use this parameter if you are only interested in "
                        "the first best decoding result. This option has a "
                        "different effect depending on the used --decoder. For"
                        " the beam decoder, it means stopping decoding when "
                        "the best active hypothesis ends with </s>. If false, "
                        "do not stop until all hypotheses end with EOS. For "
                        "the dfs and restarting decoders, early stopping "
                        "enables admissible pruning of branches when the "
                        "accumulated score already exceeded the currently best "
                        "score. DO NOT USE early stopping in combination with "
                        "the dfs or restarting decoder when your predictors "
                        "can produce positive scores!")
    group.add_argument("--gumbel", action='store_true',
                        help="Add gumbel random variable as in Kool et. al 2019. "
                        "effectively makex beam search random sampling")
    group.add_argument('--temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
    group.add_argument("--seed", default=0, type=int,
                        help="Random seed to use for numpy operations")


    ### Decoding Regularizers
    group.add_argument("--length_normalization", default=False, action="store_true",
                       help="Normalize hypothesis score by length. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    group.add_argument("--variance_regularizer", default=0., type=float,
                       help="UID variance regularizer. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    group.add_argument("--greedy_regularizer", default=0., type=float,
                       help="UID greedy regularizer. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    group.add_argument("--local_variance_regularizer", default=0., type=float,
                       help="UID local variance regularizer. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    group.add_argument("--max_regularizer", default=0., type=float,
                       help="UID max regularizer. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    group.add_argument("--square_regularizer", default=0., type=float,
                       help="UID squared regularizer. Parameter sets strength "
                       "of penalty (should be positive to encourage UID)")
    

    ## Output options
    group = parser.add_argument_group('Output options')
    group.add_argument("--nbest", default=0, type=int,
                        help="Maximum number of hypotheses in the output "
                        "files. Set to 0 to output all hypotheses found by "
                        "the decoder. If you use the beam or astar decoder, "
                        "this option is limited by the beam size.")
    group.add_argument("--num_log", default=1, type=int,
                        help="Maximum number of hypotheses to log")
    group.add_argument("--output_path", default="sgnmt-out.%s",
                        help="Path to the output files generated by SGNMT. You "
                        "can use the placeholder %%s for the format specifier")
    group.add_argument("--outputs", default="",
                        help="Comma separated list of output formats: \n\n"
                        "* 'text': First best translations in plain text "
                        "format\n"
                        "* 'nbest': Moses' n-best format with separate "
                        "scores for each predictor.\n"
                        "* 'nbest_sep': nbest translations in plain text "
                        "output to individual files based off of 'output_path'\n"
                        "* 'score': writes scores of hypotheses to file; output "
                        "is line-by-line.\n"
                        "* 'ngram': MBR-style n-gram posteriors.\n\n"
                        "For extract_scores_along_reference.py, select "
                        "one of the following output formats:\n"
                        "The path to the output files can be specified with "
                        "--output_path")
    group.add_argument("--remove_eos", default=True, type='bool',
                        help="Whether to remove </S> symbol on output.")
    group.add_argument("--src_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). See --preprocessing and --postprocessing for "
                        "more details.")
    group.add_argument("--trg_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). See --preprocessing and --postprocessing for "
                        "more details.")
    group.add_argument("--wmap", default="",
                        help="Sets --src_wmap and --trg_wmap at the same time")
    group.add_argument("--preprocessing", default="id",
                        choices=['id','word', 'char', 'bpe', 'bpe@@'],
                        help="Preprocessing strategy for source sentences.\n"
                        "* 'id': Input sentences are expected in indexed "
                        "representation (321 123 456 4444 ...).\n"
                        "* 'word': Apply --src_wmap on the input.\n"
                        "* 'char': Split into characters, then apply "
                        "(character-level) --src_wmap.\n"
                        "* 'bpe': Apply Sennrich's subword_nmt segmentation \n"
                        "SGNMT style (as in $SGNMT/scripts/subword-nmt)\n"
                        "* 'bpe@@': Apply Sennrich's subword_nmt segmentation "
                        "with original default values (removing </w>, using @@"
                        " separator)\n")
    group.add_argument("--postprocessing", default="id",
                        choices=['id','word', 'bpe@@','wmap', 'char', 'subword_nmt', 'bpe_'],
                        help="Postprocessing strategy for output sentences. "
                        "See --preprocessing for more.")
    group.add_argument("--bpe_codes", default="",
                        help="Must be set if preprocessing=bpe. Path to the "
                        "BPE codes file from Sennrich's subword_nmt.")
    group.add_argument("--add_incomplete", default=False, type='bool',
                        help="If nbest hypotheses are not found, add incomplete "
                        "hypotheses to output")
    group.add_argument("--no_statistics", default=False, action="store_true",
                       help="Do not compute or print hypothesis UID statistics. "
                       "Cannot be set if using UID decoding.")
    
    ## Predictor options
    group = parser.add_argument_group('General predictor options')
    group.add_argument("--predictor", default="fairseq",
                        help="Predictors are scoring modules which define a distribution "
                        "over target words given the history and some side "
                        "information like the source sentence. If vocabulary "
                        "sizes differ among predictors, we fill in gaps with "
                        "predictor UNK scores.:\n\n"
                        "* 'fairseq': fairseq predictor.\n"
                        "         Options: fairseq_path, fairseq_user_dir, "
                        "fairseq_lang_pair, n_cpu_threads")    
    
            
    return parser

def parse_args(parser):
    args, _ = parser.parse_known_args()
    if args.decoder is not None:
        decoding.DECODER_REGISTRY[args.decoder].add_args(parser)
    if args.predictor is not None:
        import predictors
        predictors.PREDICTOR_REGISTRY[args.predictor].add_args(parser)
    return parser.parse_args()


def get_args():
    parser = get_parser()
    args = parse_args(parser)
    return args


def validate_args(args):
    """Some rudimentary sanity checks for configuration options.
    This method directly prints help messages to the user. In case of fatal
    errors, it terminates using ``logging.fatal()``
    
    Args:
        args (object):  Configuration as returned by ``get_args``
    """
    # Validate --range
    if args.range and args.input_method == 'shell':
        logging.warn("The --range parameter can lead to unexpected "
                     "behavior in 'shell' mode.")
    
    # TODO: add one for gumbels
    # Some common pitfalls
    sanity_check_failed = False
    if args.input_method == 'dummy' and args.max_len_factor < 10:
        logging.warn("You are using the dummy input method but a low value "
                     "for max_len_factor (%d). This means that decoding will "
                     "not consider hypotheses longer than %d tokens. Consider "
                     "increasing max_len_factor to the length longest relevant"
                     " hypothesis" % (args.max_len_factor, args.max_len_factor))
        sanity_check_failed = True
    if "fairseq" in args.predictor and args.indexing_scheme != "fairseq":
        logging.warn("You are using the fairseq predictor, but indexing_scheme "
                     "is not set to fairseq.")
        sanity_check_failed = True
    if args.preprocessing != "id" and not args.wmap and not args.src_wmap:
        logging.warn("Your preprocessing method needs a source wmap.")
        sanity_check_failed = True
    if args.postprocessing != "id" and not args.wmap and not args.trg_wmap:
        logging.warn("Your postprocessing method needs a target wmap.")
        sanity_check_failed = True
    if args.no_statistics and any([args.variance_regularizer, args.square_regularizer,
                                args.max_regularizer, args.local_variance_regularizer,
                                args.greedy_regularizer]):
        logging.warn("Cannot turn off stats computation when using UID decoding")
        sanity_check_failed = True
    if (args.gumbel or 'sampling' in args.decoder):
        if not args.nbest:
            logging.warn("Must set nbest equivalent to number of desired samples "
                        "when using gumbel or sampling decoders; beam size will not be used.")
            sanity_check_failed = True
    elif args.beam == 0 :
        logging.warn("Beam size of 0 used. No hypotheses will be pruned!")

    if sanity_check_failed and not args.ignore_sanity_checks:
        raise AttributeError("Sanity check failed (see warnings). If you want "
            "to proceed despite these warnings, use --ignore_sanity_checks.")

