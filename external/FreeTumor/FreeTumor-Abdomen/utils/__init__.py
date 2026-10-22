from .data_utils import *


def get_loader(args):
    if args.data == 'lits':
        return get_loader_lits(args)

    elif args.data == 'panc':
        return get_loader_pancreas(args)

    elif args.data == 'kits':
        return get_loader_kits(args)

    elif args.data == 'all':
        return get_loader_all(args)

    elif args.data == 'syn':
        return get_loader_for_syn(args)

    else:
        print('error name')
        return None