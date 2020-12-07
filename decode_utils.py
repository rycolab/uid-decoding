import logging
import codecs
import sys
import time
import traceback
import os
import uuid

import ui
import io_utils
import utils
import decoding
import output
import predictors


args = None
"""This variable is set to the global configuration when 
base_init().
"""

def base_init(new_args):
    """This function should be called before accessing any other
    function in this module. It initializes the `args` variable on 
    which all the create_* factory functions rely on as configuration
    object, and it sets up global function pointers and variables for
    basic things like the indexing scheme, logging verbosity, etc.

    Args:
        new_args: Configuration object from the argument parser.
    """
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
    
    ui.validate_args(args)
    if args.run_diagnostics:
        ui.run_diagnostics()
        sys.exit()


def add_predictor(decoder):
    """Adds all enabled predictors to the ``decoder``. This function 
    makes heavy use of the global ``args`` which contains the
    SGNMT configuration. Particularly, it reads out ``args.predictors``
    and adds appropriate instances to ``decoder``.
    TODO: Refactor this method as it is waaaay tooooo looong
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add predictors to this instance with
            ``add_predictor()``
    """
    preds = utils.split_comma(args.predictor)
    if not preds:
        logging.fatal("Require at least one predictor! See the --predictors "
                      "argument for more information.")
    
    if len(preds) > 1:
        logging.fatal("Only 1 predictor supported at the moment")

    pred = preds[0]
    try:
        predictor = predictors.PREDICTOR_REGISTRY[pred](args)
        decoder.add_predictor(pred, predictor)
        logging.info("Initialized predictor {}".format(
                         pred))
    except IOError as e:
        logging.fatal("One of the files required for setting up the "
                      "predictors could not be read: %s" % e)
        decoder.remove_predictor()
    except AttributeError as e:
        logging.fatal("Invalid argument for one of the predictors: %s."
                       "Stack trace: %s" % (e, traceback.format_exc()))
        decoder.remove_predictors()
    except NameError as e:
        logging.fatal("Could not find external library: %s. Please make sure "
                      "that your PYTHONPATH and LD_LIBRARY_PATH contains all "
                      "paths required for the predictors. Stack trace: %s" % 
                      (e, traceback.format_exc()))
        decoder.remove_predictor()
    except ValueError as e:
        logging.fatal("A number format error occurred while configuring the "
                      "predictors: %s. Please double-check all integer- or "
                      "float-valued parameters such as --predictor_weights and"
                      " try again. Stack trace: %s" % (e, traceback.format_exc()))
        decoder.remove_predictor()
    except Exception as e:
        logging.fatal("An unexpected %s has occurred while setting up the pre"
                      "dictors: %s Stack trace: %s" % (sys.exc_info()[0],
                                                       e,
                                                       traceback.format_exc()))
        decoder.remove_predictor()


def create_decoder():
    """Creates the ``Decoder`` instance. This specifies the search 
    strategy used to traverse the space spanned by the predictors. This
    method relies on the global ``args`` variable.
    Returns:
        Decoder. Instance of the search strategy
    """
    # Create decoder instance and add predictors
    
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


def create_output_handlers():
    """Creates the output handlers defined in the ``io`` module. 
    These handlers create output files in different formats from the
    decoding results.
    
    Args:
        args: Global command line arguments.
    
    Returns:
        list. List of output handlers according --outputs
    """
    if not args.outputs:
        return []
    outputs = []
    for name in utils.split_comma(args.outputs):
        path = args.output_path % name if '%s' in args.output_path else args.output_path
        try:
            outputs.append(output.OUTPUT_REGISTRY[name](path, args))
        except KeyError:
            logging.fatal("Output format %s not available. Please double-check"
                          " the --outputs parameter." % name)
    return outputs


def get_sentence_indices(range_param, src_sentences):
    """Helper method for ``do_decode`` which returns the indices of the
    sentence to decode
    
    Args:
        range_param (string): ``--range`` parameter from config
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    ids = []
    if args.range:
        try:
            if ":" in args.range:
                from_idx,to_idx = args.range.split(":")
            else:
                from_idx = int(args.range)
                to_idx = from_idx
            ids = range(int(from_idx)-1, int(to_idx))
        except Exception as e:
            logging.info("The --range does not seem to specify a numerical "
                         "range (%s). Interpreting as file name.." % e)
            tmp_path = "%s/sgnmt-tmp.%s" % (os.path.dirname(args.range), 
                                            uuid.uuid4())
            logging.debug("Temporary range file: %s" % tmp_path)
            while True:
                try:
                    os.rename(args.range, tmp_path)
                    with open(tmp_path) as tmp_f:
                        all_ids = [i.strip() for i in tmp_f]
                    next_id = None
                    if all_ids:
                        next_id = all_ids[0]
                        all_ids = all_ids[1:]
                    with open(tmp_path, "w") as tmp_f:
                        tmp_f.write("\n".join(all_ids))
                    os.rename(tmp_path, args.range)
                    if next_id is None:
                        return
                    logging.debug("Fetched ID %s and updated %s"
                                  % (next_id, args.range))
                    yield int(next_id)-1
                except Exception as e:
                    logging.debug("Could not fetch sentence ID from %s (%s). "
                                  "Trying again in 2 seconds..." 
                                  % (args.range, e))
                    time.sleep(2)
    else:
        if src_sentences is False:
           logging.fatal("Input method dummy requires --range")
        else:
            ids = range(len(src_sentences))
    for i in ids:
        yield i


def _get_text_output_handler(output_handlers):
    """Returns the text output handler if in output_handlers, or None."""
    for output_handler in output_handlers:

        if isinstance(output_handler, output.TextOutputHandler)\
                or isinstance(output_handler, output.NBestSeparateOutputHandler):
            return output_handler
    return None

def _get_score_output_handler(output_handlers):
    """Returns the text output handler if in output_handlers, or None."""
    for output_handler in output_handlers:
        if isinstance(output_handler, output.ScoreOutputHandler):
            return output_handler
    return None


def _postprocess_complete_hypos(hypos):
    """This function applies the following operations on the list of
    complete hypotheses returned by the Decoder:

      - </s> removal
      - Apply --nbest parameter if necessary
      - Applies combination_scheme on full hypotheses, reorder list

    Args:
      hypos (list): List of complete hypotheses

    Returns:
      list. Postprocessed hypotheses.
    """
    if args.remove_eos:
        for hypo in hypos:
            if (hypo.trgt_sentence 
                    and hypo.trgt_sentence[-1] == utils.EOS_ID):
                hypo.trgt_sentence = hypo.trgt_sentence[:-1]
    if args.nbest > 0:
        hypos = hypos[:args.nbest]
    return hypos


def _generate_dummy_hypo():
    return decoding.core.Hypothesis([utils.UNK_ID], 0.0, [0.0]) 


def do_decode(decoder, 
              output_handlers, 
              src_sentences,
              trgt_sentences=None,
              num_log=1):
    """This method contains the main decoding loop. It iterates through
    ``src_sentences`` and applies ``decoder.decode()`` to each of them.
    At the end, it calls the output handlers to create output files.
    
    Args:
        decoder (Decoder):  Current decoder instance
        output_handlers (list):  List of output handlers, see
                                 ``create_output_handlers()``
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    if not decoder.has_predictor():
        logging.fatal("Terminated due to an error in the "
                      "predictor configuration.")
        return
    all_hypos = []
    text_output_handler = _get_text_output_handler(output_handlers)
    if text_output_handler:
        text_output_handler.open_file()
    score_output_handler = _get_score_output_handler(output_handlers)

    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    sen_indices = []
    diversity_metrics = []
    not_full = 0

    for sen_idx in get_sentence_indices(args.range, src_sentences):
        decoder.set_current_sen_id(sen_idx)
        try:
            src = "0" if src_sentences is False else src_sentences[sen_idx]
            if len(src.split()) > 1000:
                print("Skipping ID", str(sen_idx), ". Too long...")
                continue
            src_print = io_utils.src_sentence(src)
            logging.info("Next sentence (ID: %d): %s" % (sen_idx + 1, src_print))
            src = io_utils.encode(src)
            start_hypo_time = time.time()
            decoder.apply_predictor_count = 0
            if trgt_sentences:
                hypos = decoder.decode(src, io_utils.encode_trg(trgt_sentences[sen_idx]))
            else:
                hypos = decoder.decode(src)
            if not hypos:
                logging.error("No translation found for ID %d!" % (sen_idx+1))
                logging.info("Stats (ID: %d): score=<not-found> "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        decoder.apply_predictor_count,
                                        time.time() - start_hypo_time))
                hypos = [_generate_dummy_hypo()]
            
            hypos = _postprocess_complete_hypos(hypos)
            for logged_hypo in hypos[:num_log]:
                logging.info("Decoded (ID: %d): %s" % (
                            sen_idx+1,
                            io_utils.decode(logged_hypo.trgt_sentence)))
                logging.info("Stats (ID: %d): score=%f "
                             "num_expansions=%d "
                             "time=%.2f " 
                             "perplexity=%.2f"% (sen_idx+1,
                                            logged_hypo.total_score,
                                            decoder.apply_predictor_count,
                                            time.time() - start_hypo_time,
                                            utils.perplexity(logged_hypo.score_breakdown)))
                if logged_hypo.statistics:
                    logging.info("UID Stats (ID: %d): variance=%.3f "
                                 "local variance=%.3f "
                                 "greedy=%.3f " 
                                 "squared=%.3f "
                                 "max=%.3f"% (sen_idx+1, logged_hypo.statistics.variance(ddof=0), 
                                    logged_hypo.statistics.local_variance(ddof=0),
                                    logged_hypo.statistics.max_offset(), 
                                    logged_hypo.statistics.squares(), 
                                    -logged_hypo.statistics.minimum()))

            if score_output_handler:
                try:
                    score_output_handler.write_score(logged_hypo.score_breakdown)
                except IOError as e:
                    logging.error("I/O error %d occurred when creating output files: %s"
                                % (sys.exc_info()[0], e))

            if decoder.nbest > 1:
                diversity_score = utils.ngram_diversity([io_utils.decode(h.trgt_sentence) for h in hypos])
                logging.info("Diversity: score=%f "
                          % (diversity_score))
                diversity_metrics.append(diversity_score)

                if len(hypos) < decoder.nbest:
                    not_full += 1
            
            all_hypos.append(hypos)
            sen_indices.append(sen_idx)
            try:
                # Write text output as we go
                if text_output_handler:
                    text_output_handler.write_hypos([hypos])
            except IOError as e:
                logging.error("I/O error %d occurred when creating output files: %s"
                            % (sys.exc_info()[0], e))
        except ValueError as e:
            logging.error("Number format error at sentence id %d: %s, "
                          "Stack trace: %s" % (sen_idx+1, 
                                               e,
                                               traceback.format_exc()))
        except AttributeError as e:
            logging.fatal("Attribute error at sentence id %d: %s. This often "
                          "indicates an error in the predictor configuration "
                          "which could not be detected in initialisation. "
                          "Stack trace: %s" 
                          % (sen_idx+1, e, traceback.format_exc()))
        except Exception as e:
            logging.error("An unexpected %s error has occurred at sentence id "
                          "%d: %s, Stack trace: %s" % (sys.exc_info()[0],
                                                       sen_idx+1,
                                                       e,
                                                       traceback.format_exc()))
            try:
                # Write text output as we go
                if text_output_handler:
                    hypos = [_generate_dummy_hypo()]
                    text_output_handler.write_hypos([hypos])
            except IOError as e:
                logging.error("I/O error %d occurred when creating output files: %s"
                            % (sys.exc_info()[0], e))


    logging.info("Decoding finished. Time: %.2f" % (time.time() - start_time))
    if decoder.nbest > 1:
        print(diversity_metrics)
    print("Total not full:", str(not_full))
    try:
        for output_handler in output_handlers:
            if output_handler == text_output_handler:
                output_handler.close_file()
            else:
                output_handler.write_hypos(all_hypos, sen_indices)
    except IOError as e:
        logging.error("I/O error %s occurred when creating output files: %s"
                      % (sys.exc_info()[0], e))

