# UID Decoding
Decoding library based on SGNMT: https://github.com/ucam-smt/sgnmt. See their [docs](http://ucam-smt.github.io/sgnmt/html/) for setting up a fairseq model to work with the library.

## Dependencies and Setup
Clone library and submodules
```
git clone --recurse-submodules https://github.com/rycolab/uid-decoding.git
```

Dependencies
```
fairseq==0.10.1
scipy==1.5.4
numpy==1.19.4
Cython==0.29.21
sortedcontainers==2.3.0
subword-nmt==0.3.7

```

To compile the datastructure classes, run:
```
pip setup install -e .
```

To compile the statistics classes, navigate to the `runstats` submodule:
```
cd runstats
python setup.py install
```


## Getting Started
We recommend starting with the pretrained models available from fairseq. Download any of the models from, e.g., their NMT examples, unzip, and place model checkpoints in `data/ckpts`. You'll have to preprocess the dictionary files to a format that the library expects. Using the [pre-trained convolutional English-French WMT‘14 model](https://github.com/pytorch/fairseq/tree/master/examples/translation) an example:

```
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -
cat wmt14.en-fr.fconv-py/dict.en.txt | awk 'BEGIN{print "<epsilon> 0\n<s> 1\n</s> 2\n<unk> 3"}{print $1" "(NR+3)}' > wmap.en
cat wmt14.en-fr.fconv-py/dict.fr.txt | awk 'BEGIN{print "<epsilon> 0\n<s> 1\n</s> 2\n<unk> 3"}{print $1" "(NR+3)}' > wmap.fr

``` 

Tokenization (for input) and detokenization (for output) should be performed with the [mosesdecoder library](https://github.com/moses-smt/mosesdecoder.git). If the model uses BPE, you'll have to preprocess the input file to put words in byte pair format. Given your bpecodes listed in e.g., file `bpecodes`, the entire pre-processing of input file `input_file.txt` in English (en) can be done as follows. Again using the convolutional English-French WMT‘14 model with the [`newstest` test set](http://statmt.org/wmt14/test-full.tgz) as an example input file:

#### Remove special formatting from newstest set
```
grep '<seg id' test-full/newstest2014-fren-src.en.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > newstest_cleaned.txt
```
#### Tokenize and apply BPE
```
cat newstest_cleaned.txt | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en > out
subword-nmt apply-bpe -c wmt14.en-fr.fconv-py/bpecodes -i out -o newstest_bpe.txt
```

Alternatively, one can play around with the toy model in the test scripts. Outputs are not meaningful but it is deterministic and useful for debugging.

### Beam Search

Basic beam search can be performed on a fairseq model translating from German to English on the IWSLT dataset as follows:

```
 python decode.py  --fairseq_path wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder beam --beam 10 
 ```

A faster version, best first beam search, simply changes the decoder:

```
 python decode.py  --fairseq_path wmt14.en-fr.fconv-py/model.pt --fairseq_lang_pair en-fr --src_wmap wmap.en --trg_wmap wmap.fr --input_file
 newstest_bpe.txt --preprocessing word --postprocessing bpe@@ --decoder dijkstra_ts --beam 10 
 ```

By default, both decoders only return the best solution. Set `--early_stopping False` if you want the entire set.

A basic example of outputs can be seen when using the test suite:

 ```
 python test.py --decoder beam --beam 10 
 ```

 Additionally, you can run
 ```
 python decode.py --help
 ```
 to see descriptions of all available arguments.

### Regularizers

The following flags may be used to include UID-regularized decoding:
```
--greedy_regularizer <lambda>
--local_variance_regularizer <lambda>
--variance_regularizer <lambda>
--max_regularizer <lambda>
--square_regularizer <lambda>
```
where `<lambda>` should be a (positive) float specifying the strength of the penalty. Note that it is not recommended to use `early_stopping` with the variance or local variance regularizers.

To use length normalized scores, add the `--length_normalization` flag.


### Outputs

To see all outputs, set `--num_log <n>` for however many outputs (per input) you'd like to see. To write all outputs to files, set `--outputs nbest_sep --output_path <path_prefix>`. You'll then get a file of samples for each position (not each input!). To just write the first/best output to a file, use `--outputs text --output_path <output_file_name>`

### Scoring
 Scoring is not integrated into the library but can be performed afterwards using, e.g., the `sacrebleu` package. Make sure you use the arguments `--outputs text --output_path <output_file_name> ` during decoding and then detokenize the text using the mosesdecoder detokenizer script. Given a (detokenized) baseline, you can then run sacrebleu to calculate BLEU. For example:

 ```
 cat <output_file_name> | perl mosesdecoder/scripts/tokenizer/detokenizer.perl -threads 8 -l en | sacrebleu reference.txt
 ```

