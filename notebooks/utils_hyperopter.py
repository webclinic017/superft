from mount import utils

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_command(command: str, args: dict):
    cmd = command
    for key, value in args.items():
        cmd += f" --{key} {value}"
    return cmd


def get_hyperopt_command(
    preset: str, 
    timerange_start: str, 
    timerange_end: str, 
    hyperopt_options: dict,
    fee: float = 0.001
    ):
    user_data_dir = f"/freqtrade/mount/presets/{preset}"
    arguments = {
        "config": f"{user_data_dir}/config-backtesting.json",
        "logfile": f"{user_data_dir}/exports/hyperopt.log",
        # "export-filename": f"{user_data_dir}/exports/backtesting.json",
        "strategy-path": f"{user_data_dir}/strategies", 
        "strategy": "NotebookStrategy",
        "datadir": "/freqtrade/user_data/data/binance",
        "userdir": user_data_dir,
        "timerange": "%s-%s" % (timerange_start.replace("-", ""), timerange_end.replace("-", "")),
        # "export": "trades",
        "fee": str(fee), # Extra for spread
        "job-workers": 12,
        **hyperopt_options,
        "print-all": "",
    }
    return parse_command("freqtrade hyperopt", arguments), arguments


def post_hyperopt(preset: str, command: str, command_args: dict, stdout: str):
    """
    1. Copy presets's
    - strategies folder -> mount/backtest_history/{date}_{clock}_{preset}/ (...) strategies
    - backtesting command -> .../exports/backtesting_command.yml (From get_backtesting_command() output)
    - backtesting results -> .../exports/backtesting_result.json (From load latest backtest results as dict then save .json)
    - profit plot -> .../exports/profits.png (From plot profits function saved as image)
    - stdout results -> .../exports/stdout.log (From print stdout that stored in variable)
    - config-training.json, config-backtesting.json, config-live.json
    """
    working_path_strategies = utils.get_preset_path(preset, "strategies")
    working_path_stdout = utils.get_preset_path(preset, "exports", "hyperopt.log")
    config_filenames = ["config-training.json", "config-backtesting.json", "config-live.json"]
    hyperopt_history = "hyperopt_history"
    
    target_foldername = preset + "_" + utils.get_readable_date()
    target_path_folder = (hyperopt_history, target_foldername)

    target_path_strategies = utils.get_mount_path(*target_path_folder, "strategies")
    target_path_output = utils.get_mount_path(*target_path_folder, "exports", "hyperopt_output.yml")
    target_path_log = utils.get_mount_path(*target_path_folder, "exports", "hyperopt.log")
    target_path_result = utils.get_mount_path(*target_path_folder, "exports", "hyperopt_result.fthypt")
    
    # Make new folders
    os.mkdir(utils.get_mount_path(*target_path_folder))
    os.mkdir(utils.get_mount_path(*target_path_folder, "exports"))
    
    # Copy strategies folder
    utils.safe_copy(working_path_strategies, target_path_strategies)
    
    # Copy configs
    for config_filename in config_filenames:
        frompath = utils.get_preset_path(preset, config_filename)
        targetpath = utils.get_mount_path(*target_path_folder, config_filename)
        utils.safe_copy(frompath, targetpath)
    
    # Save command, args, and its stdout
    output_dict = {"preset": preset, "command": command, "arguments": command_args, "stdout": stdout}
    utils.write_as_yaml(output_dict, target_path_output)
    
    # Save result
    lastrun = utils.get_preset_path(preset, "hyperopt_results", ".last_result.json")
    
    with open(lastrun, "r") as f:
        last_run_filename = json.load(f)["latest_hyperopt"]
        path_lastrun = utils.get_preset_path(preset, "hyperopt_results", last_run_filename)
    
    utils.safe_copy(path_lastrun, target_path_result)
    utils.safe_copy(working_path_stdout, target_path_log)

        