# coding: utf-8

"""
This script watches `ps` output, parses it for the given process, logs the stas
for the given interval and writes it to a file after termination, so that we can
create plots or whatever from the data.
This is intended to replace the constant staring at `htop` for probing process
ressource usage.
"""

import os
import sys
import time
import json
import subprocess
import argparse


def sample_ps():
    """
    Sample and parse `ps` output for given PID.

    Returns
    -------
    out_narrow : dict
        Dictionary with header names as keys and information for all processes
        for that key as a list as values.
    out_wide : list
        List of dictionaries, one for each process, where each dict has header
        names as keys and information for the current process as values.
    """
    cmd = ["ps", "aux"]
    out = subprocess.check_output(cmd).decode("utf-8").split("\n")
    # Parse columns, be careful about CMD column which may contain extra
    # whitespaces. So split and glue and use header count to get data columns
    header = [c.strip() for c in out[0].split(None)]
    ncols = len(header)
    data = [line.split(None, maxsplit=ncols - 1)
            for line in out[1:] if line.strip()]
    for i, line in enumerate(data):
        if len(line) != ncols:
            raise ValueError(
                "Unexpected output while parsing `ps`. Line {} has not length "
                "{}: {}".format(i + 1, ncols, ", ".join(line)))
    out_narrow = {h: [d[i] for d in data] for i, h in enumerate(header)}
    out_wide = [{h: di for h, di in zip(header, d)} for d in data]
    return out_narrow, out_wide


def _save_output(output, fname):
    with open(fname, "w") as outf:
        json.dump(output, fp=outf, indent=2)
        print("Saved output to '{}'".format(fname))


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to watch.")
    parser.add_argument(
        "-o", "--outf", type=str, required=True,
        help="Filename of the output file. Data is stored in JSON format.")
    parser.add_argument(
        "-n", "--interval", type=int, default=2,
        help="Integer time interval in seconds to sample `ps`. Default: 2.")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="If given, print message for each sample.")
    args = parser.parse_args()

    # Check input args
    if args.interval < 1:
        sys.exit("Interval must be >= 1 second.")

    path = os.path.dirname(os.path.expandvars(os.path.expanduser(args.outf)))
    if not os.path.isdir(path):
        raise ValueError("Output directory '{}' does not exist.".format(path))
    fname = os.path.basename(args.outf)
    if not fname:
        raise ValueError("Given filename is invalid.")
    fname = os.path.join(path, fname)
    if not fname.endswith(".json"):
        fname += ".json"

    stats_out = {}  # Output data
    while True:
        try:
            _t0 = time.time()

            # Sample ps stats
            ps_info_narrow, ps_info_wide = sample_ps()
            # Find desired PID
            try:
                pids = ps_info_narrow["PID"]
            except ValueError as err:
                _save_output(stats_out, fname)
                sys.exit(err)
            except KeyError:
                _save_output(stats_out, fname)
                sys.exit("No PID information in parsed output.")
            try:
                idx = pids.index(args.pid)
            except ValueError:
                _save_output(stats_out, fname)
                sys.exit("Requested PID '{}' "
                         "not in `ps` output.".format(args.pid))
            # Append or init output data
            proc_stats = ps_info_wide[idx]
            if not stats_out:
                stats_out = {k: [v] for k, v in proc_stats.items()}
                start_time = _t0
                stats_out["sample_time"] = [0]  # Record relative times
            else:
                stats_out["sample_time"].append(_t0 - start_time)
                for name in proc_stats.keys():
                    stats_out[name].append(proc_stats[name])

            _dt = time.time() - _t0
            if args.verbose:
                print(
                    "{:8.1f}s : Successfully sampled `ps`, next sample "
                    "in {:.3f}s".format(
                        stats_out["sample_time"][-1], args.interval - _dt))
            time.sleep(args.interval - _dt)
        except KeyboardInterrupt:
            _save_output(stats_out, fname)
            sys.exit("Programm ended by user.")


if __name__ == "__main__":
    _main()
