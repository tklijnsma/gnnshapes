# -*- coding: utf-8 -*-
import os, shutil, logging, subprocess, glob, pprint, time, datetime, sys
import os.path as osp

# ___________________________________________________________________
# Loggers

COLORS = {
    'yellow' : '\033[33m',
    'red'    : '\033[31m',
    }
RESET = '\033[0m'

def colored(text, color=None):
    if not color is None:
        text = COLORS[color] + text + RESET
    return text

def setup_logger(name='glue', fmt=None):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.warning('Logger %s is already defined', name)
    else:
        if fmt is None:
            fmt = logging.Formatter(
                fmt = (
                    colored(
                        '%(levelname)8s:%(asctime)s:%(module)s:%(lineno)s',
                        'yellow'
                        )
                    + ' %(message)s'
                    ),
                datefmt='%Y-%m-%d %H:%M:%S'
                )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger

def setup_subprocess_logger():
    return setup_logger(
        'subprocess',
        fmt = logging.Formatter(
            fmt = colored('[%(asctime)s]:', 'red') + ' %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        )

logger = setup_logger()
subprocess_logger = setup_subprocess_logger()


# ___________________________________________________________________
# Utils

def run_multiple_commands(cmds, env=None, dry=False):
    logger.info('Sending cmds:\n{0}'.format(pprint.pformat(cmds)))
    if dry:
        logger.info('Dry mode - not running command')
        return

    process = subprocess.Popen(
        'bash',
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        env = env,
        bufsize = 1,
        close_fds = True
        )

    # Break on first error (stdin will still be written but execution will be stopped)
    process.stdin.write('set -e\n')
    process.stdin.flush()

    for cmd in cmds:
        if not(is_string(cmd)):
            cmd = ' '.join(cmd)
        if not(cmd.endswith('\n')):
            cmd += '\n'
        process.stdin.write(cmd)
        process.stdin.flush()
    process.stdin.close()

    process.stdout.flush()
    for line in iter(process.stdout.readline, ""):
        if len(line) == 0: break
        subprocess_logger.info(line.rstrip('\n'))

    process.stdout.close()
    process.wait()
    returncode = process.returncode

    if (returncode == 0):
        logger.info('Command exited with status 0 - all good')
    else:
        raise subprocess.CalledProcessError(cmd, returncode)
