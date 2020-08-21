import yaml
# from easydict import EasyDict as edict
def Config(args):

    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('{0}: {1}'.format(k, v))
    print('='*20)
    return args