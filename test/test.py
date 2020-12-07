import logging
import codecs
import sys
import time
import traceback
import os
import numpy as np
import random
import string
import collections

import utils
import sampling_utils
import decoding

from test.dummy_predictor import DummyPredictor
from ui import get_args

random.seed(0)
args = None


def base_init(new_args):
    global args
    args = new_args
    # UTF-8 support
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
        logging.warn("Library is tested with Python 3, but you are using "
                     "Python 2. Expect the unexpected or switch to >3.5.")
    # Set up logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    if args.verbosity == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbosity == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbosity == 'warn':
        logging.getLogger().setLevel(logging.WARN)
    elif args.verbosity == 'error':
        logging.getLogger().setLevel(logging.ERROR)
    # Set reserved word IDs
    utils.switch_to_fairseq_indexing()

def add_predictor(decoder):
    
    p = DummyPredictor(vocab_size=20)
    decoder.add_predictor("dummy", p)

def create_decoder():
    
    try:
        decoder = decoding.DECODER_REGISTRY[args.decoder](args)
    except Exception as e:
        logging.fatal("An %s has occurred while initializing the decoder: %s"
                      " Stack trace: %s" % (sys.exc_info()[0],
                                            e,
                                            traceback.format_exc()))
        sys.exit("Could not initialize decoder.")
        
    add_predictor(decoder)
    return decoder


def _generate_dummy_hypo():
    return decoding.core.Hypothesis([utils.UNK_ID], 0.0, [0.0]) 

def create_src_sentences(num_sentences=10, str_length=5):
    return [randomString(str_length) for i in range(num_sentences)]

def randomString(stringLength=5):
    letters = string.ascii_lowercase
    return [random.choice(letters) for i in range(stringLength)]

def do_decode(decoder, 
              src_sentences,
              trgt_sentences=None,
              num_log=1):

    all_hypos = []
    
    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    for sen_idx, src in enumerate(src_sentences):
        decoder.set_current_sen_id(sen_idx)
        logging.info("Next sentence (ID: %d): %s" % (sen_idx + 1, ''.join(src)))
        decoder.apply_predictor_count = 0
        start_hypo_time = time.time()
        hypos = decoder.decode(src)
        all_hypos.append(hypos)
        if not hypos:
            logging.error("No translation found for ID %d!" % (sen_idx+1))
            logging.info("Stats (ID: %d): score=<not-found> "
                     "num_expansions=%d "
                     "time=%.2f" % (sen_idx+1,
                                    decoder.apply_predictor_count,
                                    time.time() - start_hypo_time))
            hypos = [_generate_dummy_hypo()]
        
        for logged_hypo in sorted(hypos, reverse=True)[:num_log]:
            logging.info("Decoded (ID: %d): %s" % (
                    sen_idx+1,
                    logged_hypo.trgt_sentence))
            logging.info("Stats (ID: %d): score=%f "
                        "inc=%f "
                         "num_expansions=%d "
                         "time=%.2f " 
                         "perplexity=%.2f"% (sen_idx+1,
                                        logged_hypo.total_score,
                                        logged_hypo.base_score,
                                        decoder.apply_predictor_count,
                                        time.time() - start_hypo_time,
                                        utils.perplexity(logged_hypo.score_breakdown)))
    return all_hypos


def compare_decoders(decoder1, 
              decoder2,
              src_sentences,
              early_stopping=False,
              num_log=1):
    
    all_hypos1 = do_decode(decoder1, src_sentences, num_log=num_log)
    print("-------------------")
    all_hypos2 = do_decode(decoder2, src_sentences, num_log=num_log)

    for sentences1, sentences2 in zip(all_hypos1, all_hypos2):
        if early_stopping:
            assert max(sentences1).total_score == max(sentences2).total_score
        else:
            for sen1, sen2 in zip(sorted(sentences1), sorted(sentences2)):
                assert sen1.total_score == sen2.total_score

    logging.info("Sets returned are equal!")

def test_utils():
    """
    TODO: add more tests
    """     
    from arsenal.maths import assert_equal

    for a,b in np.random.uniform(0, 10, size=(100, 2)):

        if a < b:
            a, b = b, a

        want = np.log(a-b)
        assert_equal(want, utils.log_minus(np.log(a), np.log(b)), 'log sub')
        want = np.log(a+b)
        assert_equal(want, utils.log_add(np.log(a), np.log(b)), 'log add')


args = get_args()
base_init(args)

if not args.decoder:
    test_utils()
    exit(0)

decoder = create_decoder()
try: 
    num_sentences = int(args.range)
except:
    num_sentences = 10
    logging.warn("Range argument not valid; defaulting to 10 examples")

src_sentences = create_src_sentences(num_sentences, str_length=5)
if args.beam <= 0 and not (args.gumbel or 'sampling' in args.decoder):
    logging.warn("Using beam size <= 0. Decoding may not terminate")
if args.decoder == "dijkstra_ts" and args.memory_threshold_coef <= 0:
    args.decoder = "beam"
    decoder2 = create_decoder()
    compare_decoders(decoder, decoder2, src_sentences, args.early_stopping, num_log=args.num_log)
else:
    do_decode(decoder, src_sentences, num_log=args.num_log)


