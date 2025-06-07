"""
Mostly copy paasted from DeepSpeed launcher
"""

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser, REMAINDER

import psutil
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

PID_FILE_BASEPATH = "/tmp"


def parse_args():
    parser = ArgumentParser(
        description="Treetune distributed training launch"
        " utility that creates multiple distributed"
        " processes on a single node"
    )

    parser.add_argument(
        "--nproc", type=str, default="-1", help="The number of processes to launch"
    )

    parser.add_argument(
        "--module",
        action="store_true",
        help="Change each process to interpret the launch "
        "script as a Python module, executing with the same "
        "behavior as 'python -m'.",
    )

    parser.add_argument(
        "--no_python",
        action="store_true",
        help="Skip prepending the training script with "
        "'python' - just execute it directly.",
    )

    parser.add_argument(
        "--save_pid",
        type=int,
        default=0,
        help="main launching process pid, for internal pid tracking",
    )

    parser.add_argument(
        "--enable_each_rank_log",
        default="None",
        type=str,
        help="redirect the stdout and stderr from each rank into different log files",
    )

    parser.add_argument(
        "--bind_cores_to_rank",
        action="store_true",
        help="Bind each rank to different cores of the host. "
        "This improves host efficiency especially for CPU backend",
    )

    parser.add_argument(
        "--bind_core_list",
        type=str,
        default=None,
        help="List of cores to bind to with comma separated list of "
        "numbers and range. i.e. 1,3-5,7 => [1,3,4,5,7].  When not "
        "specified, all cores on system would be used rank binding",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


# Adapted from https://psutil.readthedocs.io/en/latest/#kill-process-tree
def terminate_process_tree(pid):
    process = psutil.Process(pid)
    children = process.children(recursive=True)
    children.append(process)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=30)
    for p in alive:
        p.kill()


def main():
    args = parse_args()
    current_env = os.environ.copy()

    for k in current_env.keys():
        if "NCCL" in k:
            logger.info(f"{k}={current_env[k]}")

    num_local_procs = args.nproc
    if "," in num_local_procs:
        num_local_procs = num_local_procs.split(",")
        num_local_procs = [int(n) for n in num_local_procs]
        curr_node_idx = int(os.environ["APP_DIST__CURR_NODE_IDX"])
        inference_nodes_start_idx = int(os.environ.get("APP_DIST__INFERENCE_NODES_START_IDX", 0))
        curr_infer_node_idx = curr_node_idx - inference_nodes_start_idx
        num_local_procs = num_local_procs[curr_infer_node_idx]
    else:
        num_local_procs = int(num_local_procs)
    logger.info(f"num_local_procs={num_local_procs}")

    # set PyTorch distributed related environmental variables
    current_env["LOCAL_SIZE"] = str(num_local_procs)

    if args.save_pid:
        print(f"launcher pid: {os.getpid()}")

    pid_file = None
    if args.save_pid:
        launcher_pid = os.getpid()
        pid_file = os.path.join(PID_FILE_BASEPATH, f"{args.save_pid}.deepspeed")
        assert not os.path.isfile(pid_file), "pid file exists but shouldn't"
        with open(pid_file, "w") as fd:
            fd.write(f"{launcher_pid}")

    processes = []
    cmd = []

    if args.enable_each_rank_log != "None":
        # prepare the log path and the file name prefix
        if os.path.isfile(args.enable_each_rank_log):
            raise ValueError(
                f"{args.enable_each_rank_log} should not be a file, it should be a directory."
            )
        if not os.path.exists(args.enable_each_rank_log):
            try:
                os.makedirs(args.enable_each_rank_log)
            except Exception as e:
                print(e)
                raise ValueError(
                    f"unable to create directory {args.enable_each_rank_log} for each rank log."
                )
        log_name_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())

    for local_proc in range(0, num_local_procs):
        # each process's rank
        local_rank = local_proc
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["WORLD_SIZE"] = str(num_local_procs)
        current_env["RANK"] = str(local_rank)

        # spawn the processes
        cmd = []
        if args.bind_cores_to_rank:
            raise NotImplementedError("Binding cores to rank is not implemented yet")
        if not args.no_python:
            cmd.append(sys.executable)
            cmd.append("-u")
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag"
                    " and the '--module' flag at the same time."
                )
        cmd.append(args.training_script)
        cmd += args.training_script_args

        if args.enable_each_rank_log != "None":
            log_file = os.path.join(
                args.enable_each_rank_log,
                f"{log_name_prefix}_localrank{local_rank}.log",
            )
            log_fd = open(log_file, "w")
            process = subprocess.Popen(
                cmd, env=current_env, stdout=log_fd, stderr=log_fd
            )
        else:
            process = subprocess.Popen(cmd, env=current_env)
        # logs the command from processes
        logger.info(f"process {process.pid} spawned with command: {cmd}")
        processes.append(process)

    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            logger.info(f"Killing subprocess {process.pid}")
            try:
                terminate_process_tree(process.pid)
            except Exception:
                pass
        if last_return_code is not None:
            logger.error(f"{cmd} exits with return code = {last_return_code}")
            sys.exit(last_return_code)
        if signum in sig_names:
            logger.info(f"Main process received {sig_names[signum]}, exiting")
        if args.save_pid:
            if os.path.isfile(pid_file):
                os.remove(pid_file)
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    logger.info(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()