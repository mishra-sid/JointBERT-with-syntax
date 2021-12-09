import collections
import copy
import os 

log_dir = 'logs'
config_list = [{'config': 'baseline', 'data': 'snips', 'lr': [1e-4, 2e-4], 'nruns': 2}]

def get_flags(cfg):
    data = cfg['data']
    config = cfg['config']
    lr = cfg['lr']
    output = cfg['output']
    exp_name = cfg['exp_name']

    flags = []
    flags += [f'--lr {lr}']

    if config == 'baseline':
        flags += [f'--input {data}.txt']

    flags += [f'--output {output}']

    return flags

experiments = collections.defaultdict(list)
for cfg in config_list:
    data, config, nruns = cfg['data'], ...
    key = f'{data}.{config}'

    for _ in range(nruns):
        i_exp = len(experiments[key])
        exp_name = f'{key}.{i_exp}'
        cfg = copy.deepcopy(cfg)
        cfg['exp_name'] = exp_name
        cfg['output'] = os.path.join(log_dir, exp_name)
        flags = get_flags(cfg)

        exp = dict(name=exp_name, flags=flags, cfg=cfg)
        experiments[key].append(exp)

for exp_list in experiments.values():
    for exp in exp_list:
        output = exp['cfg']['output']
        flags = exp['flags']

        os.system(f'mkdir -p {output}')

        with open(os.path.join(output, 'script.sh'), 'w') as f:
            f.write('python {}'.format(' '.join(flags)))

        with open('experiments.txt') as f:
            f.write(f'{output}\n')