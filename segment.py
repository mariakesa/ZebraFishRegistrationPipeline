#!/usr/bin/env python

'''
Written by Fabian Kabus.
https://github.com/thefabus
'''

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import argparse
import deepdish as dd
import numpy as np
import os
from pyprind import prog_percent
import sys

#import zebrafishframework
import zebrafish_io as io
import segmentation
import time

STD_DEV_SUFFIX = '_std_dev.h5'
ROIS_SUFFIX = '_rois.npy'
TRACES_SUFFIX = '_traces.npy'
SUFFIXES = [STD_DEV_SUFFIX, ROIS_SUFFIX, TRACES_SUFFIX]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run(args):
    if args.files:
        files = args.files
    elif args.listfile:
        with open(args.listfile) as f:
            files = list(map(str.strip, f.readlines()))
    else:
        parser.print_usage()
        sys.exit(0)

    failed = False
    for f in files:
        ext = os.path.splitext(f)
        if not os.path.exists(f):
            eprint('Error: %s does not exist.' % f)
            failed = True
        elif not (ext[1] == '.h5' or ext[1] == '.hdf5'):
            eprint('Error: %s is not a hdf5 file.' % f)
            failed = True
        elif not os.path.isfile(f):
            eprint('Error: %s is not a file.' % f)
            failed = True
    if failed:
        sys.exit(1)

    sub = None
    if args.substitute:
        split = args.substitute.split(':')
        if len(split) != 2:
            eprint('Error: invalid subsitution syntax "%s". Syntax is "replace:with".' % args.substitute)
            failed = True
        sub = split
    if args.destdir:
        if not os.path.isdir(args.destdir):
            eprint('Error: destination dir "%s" does not exist.' % args.destdir)
            failed = True
    if failed:
        sys.exit(1)

    bases = []
    shifts_fns = []
    for f in files:
        base_name = os.path.splitext(f)[0]

        remove_suffix = '_aligned'
        if base_name.endswith(remove_suffix):
            base_name = base_name[:-len(remove_suffix)]

        shifts_fn = base_name + '_shifts.npy'
        shifts_fns.append(shifts_fn)
        if not args.no_shifts and not os.path.exists(shifts_fn):
            eprint('Error: "%s" does not exist.' % shifts_fn)
            failed = True

        if args.destdir:
            base_name = os.path.join(args.destdir, os.path.basename(base_name))
            bases.append(base_name)
        elif sub:
            if base_name.find(sub[0]) == -1:
                eprint('Error: filename "%s" does not contain "%s" for substitution.' % (f, sub[0]))
                failed = True
            base_name = base_name.replace(*sub)
            bases.append(base_name)
        else:
            bases.append(base_name)
    if failed:
        sys.exit(1)

    necessary = []

    if not args.no_verbose:
        print('Arguments look good. This will be processed:')
    for f, b in zip(files, bases):
        this_necessary = not all([os.path.isfile(b + s) for s in SUFFIXES]) or args.overwrite
        necessary.append(this_necessary)

        if not args.no_verbose:
            print(('' if this_necessary else '[SKIP] ') + f)
            for suffix in SUFFIXES:
                print((' -> ' if this_necessary else '[ALREADY EXISTS] ') + '%s%s' % (b, suffix))
            print()

    necessary_files = [(f, shifts_fn, b) for f, shifts_fn, b, n in zip(files, shifts_fns, bases, necessary) if n]

    if len(necessary_files) == 0:
        print('Nothing to process.')
        sys.exit(0)

    template = segmentation.load_template()
    for f, shifts_fn, b in prog_percent(necessary_files):
        print(f)
        print('='*len(f))

        try:
            base = b
            if not args.no_shifts:
                print('Loading shifts...')
                shifts = np.load(shifts_fn)
                shift_dists = np.sqrt(np.sum(np.square(shifts), axis=1))
            print('Loading stack...')
            stack = dd.io.load(f)
            print('Computing std...')

            if not args.no_shifts:
                invalid_frames = [i for i in np.arange(np.alen(stack)) if shift_dists[i] > args.shift_threshold]
            else:
                invalid_frames = []

            valid_frames = segmentation.valid_frames(invalid_frames, length=np.alen(stack))
            std = segmentation.std(stack, valid_frames=valid_frames)
            print('Saving std...')
            io.save(base + STD_DEV_SUFFIX, std, spacing=io.SPACING_JAKOB)
            print('Finding rois...')
            rois = segmentation.find_rois_template(std, template=template)
            print('Saving rois...')
            np.save(base + ROIS_SUFFIX, rois)
            print('Getting traces...')
            traces = segmentation.get_traces(stack, rois, use_radius=5)
            print('Saving traces...')
            np.save(base + TRACES_SUFFIX, traces)

        except Exception as e:
            print('An exception occured:')
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Segment multiple aligned fish into ROIs. Three files will be created for each fish:\n
*_std_dev.nrrd: Standard deviation in each pixel over time. Frames with too high shift distance are excluded (see --shift-threshold).\n
*_rois.npy: numpy array of (x, y, z, r)-tuples. xyz are the coordinates of the ROI, r is the radius if using blob detection or 0 otherwise.\n
*_traces.npy: numpy array of sampled values at the ROIs over time. See implementation for details.\n
""")
    parser.add_argument('files', nargs='*', help='List of source files (*.h5). The _aligned suffix will be removed if present.')
    parser.add_argument('-l', '--listfile', help='File with newline separated filenames.')
    group_dest = parser.add_mutually_exclusive_group()
    group_dest.add_argument('-d', '--destdir', help='Destination directory. If none is provided, destination defaults to the source directory.')
    group_dest.add_argument('-s', '--substitute', help='Subsitute parts of the path. Syntax is replace:with. For instance -s lif_files:aligned_files.')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Don\'t skip existing files.')
    parser.add_argument('--no-verbose', default=False, action='store_true', help='Omit the listing of what will be processed.')
    parser.add_argument('--shift-threshold', metavar='max', default=300, help='Shift distance threshold at which to exclude a frame from std calculation.')
    parser.add_argument('--no-shifts', action='store_true', default=False, help='Don\'t use *_shifts.npy to exclude frames (not recommended).')
    args = parser.parse_args()
    start=time.time()
    run(args)
    end=time.time()
    print('Time taken to segemnt: ', end-start)
    os._exit(0)
