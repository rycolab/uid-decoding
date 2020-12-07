from abc import abstractmethod
import os, sys
import errno
import logging
import utils
import io_utils
import numpy as np
import codecs
from collections import defaultdict
import inspect

def _mkdir(path, name):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            logging.warn("Output %s directory '%s' already exists." 
                         % (name, path))


class OutputHandler(object):
    """Interface for output handlers. """
    
    def __init__(self):
        """ Empty constructor """
        pass
    
    @abstractmethod
    def write_hypos(self, all_hypos, sen_indices=None):
        """This method writes output files to the file system. The
        configuration parameters such as output paths should already
        have been provided via constructor arguments.
        
        Args:
            all_hypos (list): list of nbest lists of hypotheses
            sen_indices (list): List of sentence indices (0-indexed)
        
        Raises:
            IOError. If something goes wrong while writing to the disk
        """
        raise NotImplementedError


class TextOutputHandler(OutputHandler):
    """Writes the first best hypotheses to a plain text file """
    name = 'text'
    def __init__(self, path, args):
        """Creates a plain text output handler to write to ``path`` """
        super(TextOutputHandler, self).__init__()
        self.path = path
        
    def write_hypos(self, all_hypos, sen_indices=None):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        if self.f is not None:
            for hypos in all_hypos:
                self.f.write(io_utils.decode(hypos[0].trgt_sentence))
                self.f.write("\n")
                self.f.flush()
        else:
            with codecs.open(self.path, "w", encoding='utf-8') as f:
                for hypos in all_hypos:
                    f.write(io_utils.decode(hypos[0].trgt_sentence))
                    f.write("\n")
                    self.f.flush()

    def open_file(self):
        self.f = codecs.open(self.path, "w", encoding='utf-8')

    def close_file(self):
        self.f.close()

class ScoreOutputHandler(OutputHandler):
    """Writes the first best hypotheses to a plain text file """
    name = 'score'
    def __init__(self, path, args):
        """Creates a plain text output handler to write to ``path`` """
        super(ScoreOutputHandler, self).__init__()
        self.path = path
        self.open_file()
        
    def write_score(self, score):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        def write(f_, score):
            f_.write(str([s[0][0] for s in score]))
            f_.write("\n")
            f_.flush()

        if self.f is not None:
            write(self.f, score)
        else:
            with codecs.open(self.path, "w", encoding='utf-8') as f:
                write(f, score)

    def write_hypos(self, all_hypos, sen_indices=None):
        pass

    def open_file(self):
        self.f = codecs.open(self.path, "w", encoding='utf-8')

    def close_file(self):
        self.f.close()


class NBestSeparateOutputHandler(OutputHandler):
    """Produces n-best files with hypotheses at respecitve positions
    """
    name = 'nbest_sep'
    def __init__(self, path, args):
        """
        Args:
            path (string):  Path to the n-best file to write
            N: n-best 
        """
        super(NBestSeparateOutputHandler, self).__init__()
        self.paths = [path + '_' + str(i) + '.txt' for i in range(max(args.nbest,1))]
        
    def write_hypos(self, all_hypos, sen_indices=None):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        if not self.f:
            self.open_file()
        for hypos in all_hypos:
            while len(hypos) < len(self.f):
                hypos.append(hypos[-1])
            for i in range(len(self.f)):
                self.f[i].write(io_utils.decode(hypos[i].trgt_sentence))
                self.f[i].write("\n")
                self.f[i].flush()

    def open_file(self):
        self.f = []
        for p in self.paths:
            self.f.append(codecs.open(p, "w", encoding='utf-8'))

    def close_file(self):
        for f in self.f:
            f.close()


class NBestOutputHandler(OutputHandler):
    """Produces a n-best file in Moses format. The third part of each 
    entry is used to store the separated unnormalized predictor scores.
    Note that the sentence IDs are shifted: Moses n-best files start 
    with the index 0, but in SGNMT and HiFST we usually refer to the 
    first sentence with 1 (e.g. in lattice directories or --range)
    """
    name = 'nbest'
    def __init__(self, path, args):
        """Creates a Moses n-best list output handler.
        
        Args:
            path (string):  Path to the n-best file to write
            predictor_names: Names of the predictors whose scores
                             should be included in the score breakdown
                             in the n-best list
        """
        super(NBestOutputHandler, self).__init__()
        self.path = path
        self.predictor_names = []
        name_count = {}
        for name in utils.split_comma(args.predictors):
            if not name in name_count:
                name_count[name] = 1
                final_name = name
            else:
                name_count[name] += 1
                final_name = "%s%d" % (name, name_count[name])
            self.predictor_names.append(final_name.replace("_", "0"))
        
        
    def write_hypos(self, all_hypos, sen_indices):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        with codecs.open(self.path, "w", encoding='utf-8') as f:
            n_predictors = len(self.predictor_names)
            for idx, hypos in zip(sen_indices, all_hypos):
                for hypo in hypos:
                    f.write("%d ||| %s ||| %s ||| %f" %
                            (idx,
                             io_utils.decode(hypo.trgt_sentence),
                             ' '.join("%s= %f" % (
                                  self.predictor_names[i],
                                  sum([s[i][0] for s in hypo.score_breakdown]))
                                      for i in range(n_predictors)),
                             hypo.total_score))
                    f.write("\n")
                idx += 1


class NgramOutputHandler(OutputHandler):
    """This output handler extracts MBR-style ngram posteriors from the 
    hypotheses returned by the decoder. The hypothesis scores are assumed to
    be loglikelihoods, which we renormalize to make sure that we operate on a
    valid distribution. The scores produced by the output handler are 
    probabilities of an ngram being in the translation.
    """
    name = 'ngram'
    def __init__(self, path, args):
        """Creates an ngram output handler.
        
        Args:
            path (string):  Path to the ngram directory to create
            min_order (int):  Minimum order of extracted ngrams
            max_order (int):  Maximum order of extracted ngrams
        """
        super(NgramOutputHandler, self).__init__()
        self.path = path
        self.min_order = args.min_order
        self.max_order = args.max_order
        self.file_pattern = path + "/%d.txt" 
      
    def write_hypos(self, all_hypos, sen_indices):
        """Writes ngram files for each sentence in ``all_hypos``.
        
        Args:
            all_hypos (list): list of nbest lists of hypotheses
            sen_indices (list): List of sentence indices (0-indexed)
        
        Raises:
            OSError. If the directory could not be created
            IOError. If something goes wrong while writing to the disk
        """
        _mkdir(self.path, "ngram")
        for sen_idx, hypos in zip(sen_indices, all_hypos):
            sen_idx += 1
            total = utils.log_sum([hypo.total_score for hypo in hypos])
            normed_scores = [hypo.total_score - total for hypo in hypos]
            ngrams = defaultdict(dict)
            # Collect ngrams
            for hypo_idx, hypo in enumerate(hypos):
                sen_eos = [utils.GO_ID] + hypo.trgt_sentence + [utils.EOS_ID]
                for pos in range(1, len(sen_eos) + 1):
                    hist = sen_eos[:pos]
                    for order in range(self.min_order, self.max_order + 1):
                        ngram = ' '.join(map(str, hist[-order:]))
                        ngrams[ngram][hypo_idx] = True
            with open(self.file_pattern % sen_idx, "w") as f:
                for ngram, hypo_indices in ngrams.items():
                    ngram_score = np.exp(utils.log_sum(
                       [normed_scores[hypo_idx] for hypo_idx in hypo_indices]))
                    f.write("%s : %f\n" % (ngram, min(1.0, ngram_score)))

OUTPUT_REGISTRY = {}

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, _cls in clsmembers:
    if issubclass(_cls, OutputHandler) and not _cls == OutputHandler:
        if not hasattr(_cls, 'name'):
            raise ValueError("All output handlers classes must have `name` attribute. Culprit: {}".format(name))
        else:
            OUTPUT_REGISTRY[_cls.name] = _cls

