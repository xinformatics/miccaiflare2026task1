from .data_utils import *


def get_loader(args):
    if args.data == 'covid':
        return get_loader_covid(args)

    else:
        print('error name')
        return None