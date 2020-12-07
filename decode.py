from __future__ import absolute_import
import logging
import os
import sys
from cmd import Cmd
import time

import utils
import io_utils
import decode_utils
from ui import get_args, get_parser, run_diagnostics

# Load configuration from command line arguments or configuration file
args = get_args()
decode_utils.base_init(args)


class SGNMTPrompt(Cmd):

    def default(self, cmd_args):
        """Translate a single sentence."""
        decode_utils.do_decode(
            decoder, outputs,
            [cmd_args.strip()])

    def emptyline(self):
        pass

    def do_translate(self, cmd_args):
        """Translate a single sentence."""
        decode_utils.do_decode(
            decoder, outputs,
            [cmd_args.strip()])

    def do_diagnostics(self, cmd_args):
        """Run diagnostics to check which external libraries are
        available to SGNMT."""
        run_diagnostics()

    def do_config(self, cmd_args):
        """Change SGNMT configuration. Syntax: 'config <key> <value>.
        For most configuration changes the decoder needs to be
        rebuilt.
        """
        global outputs, decoder, args
        split_args = cmd_args.split()
        if len(split_args) < 2:
            print("Syntax: 'config <key> <new-value>'")
        else:
            key, val = (split_args[0], ' '.join(split_args[1:]))
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    if val == "true":
                        val = True
                    elif val == "false":
                        val = False
            setattr(args, key, val)
            print("Setting %s=%s..." % (key, val))
            outputs = decode_utils.create_output_handlers()
            if key in ["wmap", "src_wmap", "trg_wmap", 
                       "preprocessing", "postprocessing", "bpe_codes"]:
                io_utils.initialize(args)
            elif not key in ['outputs', 'output_path']:
                decoder = decode_utils.create_decoder()

    def do_quit(self, cmd_args):
        """Quits SGNMT."""
        raise SystemExit

    def do_EOF(self, line):
        "Quits SGNMT"
        print("quit")
        return True


io_utils.initialize(args)
decoder = decode_utils.create_decoder()
outputs = decode_utils.create_output_handlers()

if args.input_method == 'file':
    if os.access(args.input_file, os.R_OK):
        trgt = None
        if args.trgt_file  and os.access(args.trgt_file, os.R_OK):
            with open(args.trgt_file) as f:
                trgt = [line.strip() for line in f]
        print(time.time())
        with open(args.input_file) as f:
            decode_utils.do_decode(decoder,
                                   outputs,
                                   [line.strip() for line in f],
                                   trgt,
                                   args.num_log)
        print(time.time())
    else:
        logging.fatal("Input file '%s' not readable. Please double-check the "
                      "input_file option or choose an alternative input_method."
                      % args.input_file)
elif args.input_method == 'dummy':
    decode_utils.do_decode(decoder, outputs, False)
elif args.input_method == "stdin":
    decode_utils.do_decode(decoder,
                           outputs,
                           [line.strip() for line in sys.stdin])
else: # Interactive mode: shell
    print("Starting interactive mode...")
    print("PID: %d" % os.getpid())
    print("Display help with 'help'")
    print("Quit with ctrl-d or 'quit'")
    prompt = SGNMTPrompt()
    prompt.prompt = "sgnmt> "
    prompt.cmdloop()

