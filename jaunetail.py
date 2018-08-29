import inspect
import os
import re
import json
import subprocess
from collections import defaultdict, namedtuple

# Easy object to store argument.
Argument = namedtuple('Argument', ['short_arg', 'long_arg',
                                   'takes_arg_value', 'default_when_set',
                                   'default_when_unset', 'description'])


def read_marian_help(version):
    # Initialize an empty parameter.
    _parameters = defaultdict(dict)
    k = None
    # Start reading the file.
    with open(f'marian-{version}.help') as fin:
        for line in fin:
            if not line.strip(): # Empty lines.
                continue
            if line.strip() in option_sets[version]:
                current_option_set = line.strip()[:-len(' options:')]
            else:
                if line.strip().startswith('-'):
                    # Silly exception(s)...
                    if k == '--no-reload' and line.strip() == '--model arg':
                        _parameters[current_option_set][k].append(line.strip())
                    # Replace multiple space with tab.
                    # Split on tab and assume the first item
                    # as the parameter.
                    k, *v = re.sub('\s{2,}', '\t', line.strip()).split('\t')
                    _parameters[current_option_set][k] = v
                else:
                    if k:
                        _parameters[current_option_set][k].append(line.strip())

    # Create actual dict of possible option set, `s`
    # each with their respective list of `Argument` objects.
    marian_parameters = defaultdict(list)
    # Cleaning up and extracting the parameters proper.
    # See https://regex101.com/r/3CplWT/5/ for regex pattern explanation.
    pattern = r"(-[\w]\s)?((?:\[\s)?--[\w\-]+(?:\s\])?)(\sarg\s)?(\s\[\=arg\(\=[\.\/\s\+\w-]+\)\]\s)?(\(\=[\.\/\s\+\w-]+\))?"
    for s in _parameters:
        for k in _parameters[s]:
            short_arg, long_arg, takes_arg_value, default_when_set, default_when_unset = re.match(pattern, k).groups()
            # Clean up values.
            long_arg = long_arg.strip('[| |]')
            takes_arg_value = True if takes_arg_value else False
            default_when_set = re.findall(r"\(\=([\.\/\s\+\w-]+)\)", default_when_set) if default_when_set else None
            default_when_unset = re.findall(r"\(\=([\.\/\s\+\w-]+)\)", default_when_unset) if default_when_unset else None
            description = ' '.join(_parameters[s][k])
            # Create the argyment object.
            arg = Argument(short_arg, long_arg, takes_arg_value, default_when_set, default_when_unset, description)
            marian_parameters[s].append(arg)

    return marian_parameters



option_sets = {'1.6.0': ['General options:',  'Model options:',
                         'Training options:', 'Validation set options:']}

version = '1.6.0'
