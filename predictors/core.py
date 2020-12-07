from abc import abstractmethod

import utils

class Predictor(object):
    """A predictor produces the predictive probability distribution of
    the next word given the state of the predictor. The state may 
    change during ``predict_next()`` and ``consume()``. The functions
    ``get_state()`` and ``set_state()`` can be used for non-greedy 
    decoding. Note: The state describes the predictor with the current
    history. It does not encapsulate the current source sentence, i.e. 
    you cannot recover a predictor state if ``initialize()`` was called
    in between. ``predict_next()`` and ``consume()`` must be called 
    alternately. This holds even when using ``get_state()`` and 
    ``set_state()``: Loading/saving states is transparent to the
    predictor instance.
    """
    
    def __init__(self):
        """Initializes ``current_sen_id`` with 0. """
        super(Predictor, self).__init__()
        self.current_sen_id = 0

    @staticmethod
    def add_args(parser):
        pass

    def set_current_sen_id(self, cur_sen_id):
        """This function is called between ``initialize()`` calls to 
        increment the sentence id counter. It can also be used to skip 
        sentences for the --range argument.
        
        Args:
            cur_sen_id (int):  Sentence id for the next call of
                               ``initialize()``
        """
        self.current_sen_id = cur_sen_id
    
    @abstractmethod
    def predict_next(self):
        """Returns the predictive distribution over the target 
        vocabulary for the next word given the predictor state. Note 
        that the prediction itself can change the state of the 
        predictor. For example, the neural predictor updates the 
        decoder network state and its attention to predict the next 
        word. Two calls of ``predict_next()`` must be separated by a 
        ``consume()`` call.
        
        Returns:
            dictionary,array,list. Word log probabilities for the next 
            target token. All ids which are not set are assumed to have
            probability ``get_unk_probability()``
        """
        raise NotImplementedError
    
    @abstractmethod
    def consume(self, word):
        """Expand the current history by ``word`` and update the 
        internal predictor state accordingly. Two calls of ``consume()``
        must be separated by a ``predict_next()`` call.
        
        Args:
            word (int):  Word to add to the current history
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self):
        """Get the current predictor state. The state can be any object
        or tuple of objects which makes it possible to return to the
        predictor state with the current history.
        
        Returns:
          object. Predictor state
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_state(self, state):
        """Loads a predictor state from an object created with 
        ``get_state()``. Note that this does not copy the argument but
        just references the given state. If ``state`` is going to be
        used in the future to return to that point again, you should
        copy the state with ``copy.deepcopy()`` before.
        
        Args:
           state (object): Predictor state as returned by 
                           ``get_state()``
        """
        raise NotImplementedError

    @abstractmethod
    def coalesce_and_set_states(self, states):
        """Loads a predictor state from an object created with 
        ``get_state()``. Note that this does not copy the argument but
        just references the given state. If ``state`` is going to be
        used in the future to return to that point again, you should
        copy the state with ``copy.deepcopy()`` before.
        
        Args:
           state (object): Predictor state as returned by 
                           ``get_state()``
        """
        raise NotImplementedError
    
    
    def get_unk_probability(self, posterior):
        """This function defines the probability of all words which are
        not in ``posterior``. This is usually used to combine open and
        closed vocabulary predictors. The argument ``posterior`` should 
        have been produced with ``predict_next()``
        
        Args:
            posterior (list,array,dict): Return value of the last call
                                         of ``predict_next``
        
        Returns:
            float: Score to use for words outside ``posterior``
        """
        return utils.NEG_INF

    def get_empty_str_prob(self):
        return utils.NEG_INF
    
    def initialize(self, src_sentence):
        """Initialize the predictor with the given source sentence. 
        This resets the internal predictor state and loads everything 
        which is constant throughout the processing of a single source
        sentence. For example, the NMT decoder runs the encoder network
        and stores the source annotations.
        
        Args:
            src_sentence (list): List of word IDs which form the source
                                 sentence without <S> or </S>
        """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true if two predictor states are equal, i.e. both
        states will always result in the same scores. This is used for
        hypothesis recombination
        
        Args:
            state1 (object): First predictor state
            state2 (object): Second predictor state
        
        Returns:
            bool. True if both states are equal, false if not
        """
        return False
