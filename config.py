#!usr/bin/env python
# coding:utf8

# Copyright (c) 2018, Tencent. All rights reserved

import json


class Config(object):
    """Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)
