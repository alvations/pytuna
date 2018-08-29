import inspect
import os
import re
import json
import subprocess
from collections import defaultdict, namedtuple


def autoinit(decorated_init):
    def _wrap(*args, **kwargs):
        obj,  nargs= args[0], args[1:]
        names = decorated_init.__code__.co_varnames[1:len(nargs)+1]
        nargs = {k: nargs[names.index(k)] for k in names}
        kwa_keys = decorated_init.__code__.co_varnames[len(nargs)+1:]
        kwa_defaults = inspect.getfullargspec(decorated_init).defaults
        for k in kwa_keys:
            nargs[k] = kwargs[k] if (k in kwargs) else kwa_defaults[kwa_keys.index(k)]
        for k, v in nargs.items():
            setattr(obj, k, v)
        return decorated_init(*args, **kwargs)
    return _wrap


class Transformer:
    @autoinit
    def __init__(self, marian_bin, model,
        train_src, train_trg, valid_src, valid_trg, model_type='transformer',
        vocab_src=None, vocab_trg=None, dim_vocabs=100000, max_length=100,
        mini_batch_fit=True, mini_batch=1000, maxi_batch=1000,
        valid_freq=5000, save_freq=5000, disp_freq=500,
        valid_metrics="ce-mean-words perplexity", valid_mini_batch=16,
        early_stopping=5, cost_type="ce-mean-words",
        beam_size=6, normalize=0.6,
        overwrite=True, keep_best=True,
        log=None, valid_log=None,
        enc_depth=6, dec_depth=6,
        transformer_preprocess="n", transformer_postprocess="da",
        tied_embeddings_all=0.1, dim_emb=1024, transformer_dim_ffn=4096,
        transformer_dropout=0.1, transformer_dropout_attention=0.1,
        transformer_dropout_ffn=0.1, label_smoothing=0.1,
        learn_rate=0.0001, lr_warmup=8000, lr_decay_inv_sqrt=8000,
        lr_report=True, optimizer_params="0.9 0.98 1e-09", clip_norm=5,
        devices=None, sync_sgd=True, seed=0, exponential_smoothing=True):

        # Saves the vocab in the model_dir if not specified.
        self.vocab_src = self.vocab_src if self.vocab_src else os.path.join(self.model, 'vocab.src.yml')
        self.vocab_trg = self.vocab_trg if self.vocab_trg else os.path.join(self.model, 'vocab.trg.yml')
        # Combine the vocab_src and vocab_trg into vocabs.
        self.vocabs = ' '.join([self.vocab_src, self.vocab_trg])
        del self.__dict__['vocab_src']
        del self.__dict__['vocab_trg']
        # Combine the train_src and train_trg into train_set.
        self.train_sets = ' '.join([self.train_src, self.train_trg])
        del self.__dict__['train_src']
        del self.__dict__['train_trg']
        # Combine the train_src and train_trg into train_set.
        self.valid_sets = ' '.join([self.valid_src, self.valid_trg])
        del self.__dict__['valid_src']
        del self.__dict__['valid_trg']


        # Saves the log files in the model_dir if not specified.
        self.log = self.log if self.log else os.path.join(self.model, 'train.log')
        self.valid_log = self.valid_log if self.valid_log else os.path.join(self.model, 'valid.log')
        # Set all devices if not specified
        self.devices = self.devices if self.devices else "$(seq 0 $(($(nvidia-smi -q | grep '^GPU' | wc -l) - 1)))"


    def generate_config(self):
        command_line_str = []
        command_line_str.append(self.marian_bin+ " \\")
        for argument in dir(self):
            # Get the actual argument name and value.
            arg_value = getattr(self, argument)
            arg_name = '--'+argument.replace('_', '-')
            # If it's the modeltype.
            if argument == 'model_type':
                command_line_str.append(' '.join(['--type', str(arg_value), "\\"]))
                continue
            # Skip the marian_bin:
            if argument == 'marian_bin':
                continue
            # Skip the dunder attributes.
            if argument.startswith('__') and argument.endswith('__'):
                continue
            # Skip functions.
            if hasattr(arg_value, '__call__'):
                continue
            if type(arg_value) == bool:
                if arg_value == False:
                    continue
                else:
                    command_line_str.append(arg_name+" \\")
            else:
                command_line_str.append(' '.join([arg_name, str(arg_value), "\\"]))
        command_line_str.append('--dump-config')
        return subprocess.Popen('\n'.join(command_line_str),
                                shell=True, stdout=subprocess.PIPE).stdout.read().decode('utf8')


x = Transformer('/home/ltan/marian/build/marian', 'mymodel',
                'data/train.src', 'data/train.trg', 'data/valid.src', 'data/valid.trg')

with open('config.yml', 'w') as fout:
    fout.write(x.generate_config())
