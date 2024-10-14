if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import numpy as np
import collections
import pickle
from equi_diffpo.common.robomimic_util import RobomimicObsConverter

multiprocessing.set_start_method('spawn', force=True)

def worker(x):
    path, idx = x
    converter = RobomimicObsConverter(path)
    obss = converter.convert_idx(idx)
    return obss

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()
    
    converter = RobomimicObsConverter(input)

    # save output
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # run
    idx = 0
    while idx < len(converter):
        with multiprocessing.Pool(num_workers) as pool:
            end = min(idx + num_workers, len(converter))
            results = pool.map(worker, [(input, i) for i in range(idx, end)])

        # modify action
        print('Writing {} to {}'.format(idx, end))
        with h5py.File(output, 'r+') as out_file:
            for i in tqdm(range(idx, end), desc="Writing to output"):
                obss = results[i - idx]
                demo = out_file[f'data/demo_{i}']
                del demo['obs']
                for k in obss:
                    demo.create_dataset("obs/{}".format(k), data=np.array(obss[k]), compression="gzip")
        
        idx = end
        del results


if __name__ == "__main__":
    main()
