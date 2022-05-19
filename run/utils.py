import os
from argparse import ArgumentParser, Namespace
from pprint import pprint

import yaml


def parse_yaml(parser: ArgumentParser) -> Namespace:
    parser.add_argument('-y', '--config', help="configuration file *.yml", type=str, required=False)
    args = parser.parse_args()
    # yaml priority is higher than args
    if isinstance(args.config, str) and os.path.exists(args.config):
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args_dict = args.__dict__
        args_dict.update(opt)
        args = Namespace(**args_dict)

        print("Configs:")
        pprint(args.__dict__)
        print("\n\n\n")

    return args
