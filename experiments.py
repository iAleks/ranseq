#!/usr/bin/env python2
# an experiment is composed of a list of "setting groups"
# the groups in an experiment are run in order,
# which allows later groups to override earlier groups.

# all vars must have default values set in hyperparams.py

import pretrained_nets as pret

############## choose an experiment ##############

# current = 'train'
# mod = '"reg005_nobn_onerelu"'
# current = 'finaltrain'
# mod = '"x"'
current = 'test'
mod = '"x"'

############## set up experiments ##############
exps = {}

exps['train'] = ['train',
                 'fold1',
                 '5k_iters',
                 'drop',
                 'dropi',
                 'lr4',
                 'B4',
                 'resume',
                 'fast_logging']
exps['finaltrain'] = ['train',
                      'fold99',
                      '4k_iters',
                      'drop',
                      'dropi',
                      'lr4',
                      'B4',
                      'onlyval',
                      'resume',
                      'fast_logging']
                           
exps['test'] = ['total_init',
                'testset',
                'B1',
                'never_logging']
# 'noshuf',

############## set up groups ##############
groups = {}

groups['train'] = ['do_train = True']
groups['noshuf'] = ['shuffle_train = False',
                    'shuffle_val = False']
groups['train'] = ['do_train = True']
groups['dropi'] = ['do_dropout_input = True']
groups['drop'] = ['do_dropout = True']
groups['fold0'] = ['fold = 0']
groups['fold1'] = ['fold = 1']
groups['fold2'] = ['fold = 2']
groups['fold3'] = ['fold = 3']
groups['fold4'] = ['fold = 4']
groups['fold5'] = ['fold = 5']
groups['fold6'] = ['fold = 6']
groups['fold7'] = ['fold = 7']
groups['fold8'] = ['fold = 8']
groups['fold9'] = ['fold = 9']
groups['fold99'] = ['fold = 99']
groups['testset'] = ['testset = True']
groups['debug'] = ['do_debug = True']
groups['profile'] = ['do_profile = True']
groups['B1'] = ['B = 1']
groups['B2'] = ['B = 2']
groups['B3'] = ['B = 3']
groups['B4'] = ['B = 4']
groups['B8'] = ['B = 8']
groups['B16'] = ['B = 16']
groups['B32'] = ['B = 32']
groups['B64'] = ['B = 64']
groups['B128'] = ['B = 128']
groups['B184'] = ['B = 184']
groups['lr3'] = ['lr = 1e-3']
groups['lr4'] = ['lr = 1e-4']
groups['lr45'] = ['lr = 5e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr55'] = ['lr = 5e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['1_iters'] = ['max_iters = 1']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['10_iters'] = ['max_iters = 10']
groups['50_iters'] = ['max_iters = 50']
groups['100_iters'] = ['max_iters = 100']
groups['200_iters'] = ['max_iters = 200']
groups['500_iters'] = ['max_iters = 500']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['3k_iters'] = ['max_iters = 3000']
groups['4k_iters'] = ['max_iters = 4000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['40k_iters'] = ['max_iters = 40000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['500k_iters'] = ['max_iters = 500000']
groups['resume'] = ['do_resume = True']

groups['onlyval'] = ['trainset = valset']
groups['onlytrain'] = ['valset = trainset']

groups['fastest_logging'] = ['log_freq_t = 1',
                             'log_freq_v = 1']
groups['faster_logging'] = ['log_freq_t = 10',
                            'log_freq_v = 10']
groups['fast_logging'] = ['log_freq_t = 10',
                          'log_freq_v = 10']
groups['slow_logging'] = ['log_freq_t = 100',
                          'log_freq_v = 100']
groups['never_logging'] = ['log_freq_t = 1000000000',
                           'log_freq_v = 1000000000']
groups['total_init'] = ['total_init = "' + pret.init + '"']

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print current
assert current in exps
for group in exps[current]:
    print "  " + group
    assert group in groups
    for s in groups[group]:
        print "    " + s
        _verify_(s)
        exec(s) 

s = "mod = " + mod
_verify_(s)
exec(s)
