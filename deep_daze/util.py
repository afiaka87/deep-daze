import os
import subprocess
import sys

import torch
from torch.nn import functional


def _log(msg, debug=False, to_file=sys.stdout):
    if debug is True:
        print(msg, file=to_file)


def signal_handling(signum, frame):
    global terminate
    terminate = True


def exists(val):
    return val is not None


def interpolate(image, size):
    return functional.interpolate(image, (size, size), mode='bilinear', align_corners=False)


def rand_cutout(image, size):
    width = image.shape[-1]
    offset_x = torch.randint(0, width - size, ())
    offset_y = torch.randint(0, width - size, ())
    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout


def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/', '\\')]
    if cmd_list is None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass
