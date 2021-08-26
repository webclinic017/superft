import argparse

parser = argparse.ArgumentParser(description='Free, open source crypto trading bot')
args = parser.parse_args(["freqtrade", "backtesting", "--timeperiod", "20210101"])
print(args)