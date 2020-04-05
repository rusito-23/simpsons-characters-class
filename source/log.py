#!/usr/bin/env python
# coding: utf-8

import logging
import sys


def config_logger():
    logFormatter = logging.Formatter('%(asctime)s '
                                     '[%(levelname)s] '
                                     '%(message)s')
    rootLogger = logging.getLogger('simpsons')

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.DEBUG)
