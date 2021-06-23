import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from tqdm import tqdm

from typing import List, Optional

from remapper.basecall import create_file_queue, start_producers, GuppyRead
from remapper.remapper import Remapper
from utils.bed_processing import bed_filter_factory, extract_bed_positions
from utils.models import *
from utils.writer import BinaryWriter


def get_info_args(args: argparse.Namespace) -> Dict[str, Any]:
    """ Function that extracts arguments relevant for data generation.

    This function extracts relevant arguments for data generation and optionally label (if label is defined at data
    level).

    :param args: Script arguments
    :return: Key-value pair of arguments relevant for data generation and optionally defined label
    """
    info_args = {
        'norm_method': args.norm_method,
        'mapq': args.mapq,
        'motif': args.motif,
        'index': args.index,
        'window': args.window,
        'label': args.label,
        'del_method': args.del_method
    }

    return info_args


def get_raw_signal(read: GuppyRead, norm_method, continuous: bool=True) -> np.ndarray:
    """ Returns the raw signal for the read.

    Returns the raw signal for the given read. Optionally, it converts discrete signal to continuous signal.

    :param read: Basecalled read
    :param norm_method: Signal normalization method
    :param continuous: True if returned signal should be continuous, otherwise False
    :return: Raw signal for the given read
    """
    signal = read.raw_data

    if continuous:
        signal = read.daq_scaling * (signal + read.daq_offset)

    norm_func = normalization_factory(norm_method)
    if norm_func:
        signal = norm_func(signal)

    return signal


def generate_data(read: GuppyRead,
                  reseg_data: List[ResegmentationData],
                  norm_method: str) -> Optional[List[Example]]:
    """ Function that generates data for the specified read.

    This function generates list of examples for the given read. Data is generated from raw signal for the given read,
    and information extracted from resegmentation data.

    :param read: Read for which examples will be generated
    :param reseg_data: Resegmentation data
    :param norm_method: Signal normalization method
    :return: List of generated examples for the given read and resegmentation data
    """
    signal = get_raw_signal(read, norm_method)
    all_examples = []

    for reseg_example in reseg_data:
        example_points = []

        for event_interval in reseg_example.event_intervals:
            points = signal[event_interval.start: event_interval.end]
            example_points.append(points)

        example_points = np.concatenate(example_points)

        example = Example(reseg_example.position, reseg_example.bases, example_points, reseg_example.event_lens)
        all_examples.append(example)

    if len(all_examples) == 0:
        return None

    return all_examples


def process_read(read_queue: mp.Queue, processed_queue: Optional[mp.Queue],
                 remapper: Remapper, norm_method: str,
                 label: Optional[int], bed_data: Optional[BEDData],
                 data_path: str, header_path: int) -> Optional[FeaturesData]:
    """ This function processes the given read to generate data.

    This function extracts resegmentation information, and extracts alignment data.

    :param read_queue: Queue containing basecalled reads
    :param processed_queue: Queue containing processed reads
    :param remapper: Instance of the Remapper class
    :param norm_method: Signal normalization method
    :param label: Label is 0 for unmodified reads, and 1 for modified ones
    :param bed_data: Positions used for filtering motif positions
    :param data_path: Path to the generated output data
    :param header_path: Path to the generated header
    :return: FeaturesData object if at least one example is present, otherwise None
    """
    with BinaryWriter(str(data_path), str(header_path)) as writer:
        while True:
            basecall_data = read_queue.get()
            if basecall_data is None:
                break

            # print(basecall_data.read.read_id, "- basecalled")

            remapper_data = remapper.process(basecall_data)
            if not remapper_data:
                # print(basecall_data.read.read_id, "- reseg_data is None")
                continue
            resegmentation_data, align_data = remapper_data

            # print(basecall_data.read.read_id, "- resegmented")

            examples = generate_data(basecall_data.read, resegmentation_data, norm_method)
            if not examples:
                continue

            strand = Strand.strand_from_str('+' if align_data.strand == 1 else '-')
            result = FeaturesData(align_data.ctg, strand, basecall_data.read.read_id, examples)

            if result is not None:
                try:
                    writer.write_data(result, bed_data[0] if bed_data is not None else None, label)

                except Exception as e:
                    error_callback(basecall_data.read.read_id, e)
                    # error_files += 1

            # print(basecall_data.read.read_id, "- written")


def error_callback(path, exception):
    print(f'Error for file: {path}.', file=sys.stderr)
    print(str(exception), file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)


def worker_process_reads(read_queue: mp.Queue, processed_queue: mp.Queue, reference: str,
                         norm_method: str, mapq: int, motif: str, index: int, window: int,
                         label: Optional[int], del_method: str, bed_data: Optional[BEDData],
                         output_path: str, n_processors: int) -> List[mp.Process]:
    processors = []

    remapper = Remapper(reference, mapq, motif, index, window, del_method,
                        bed_data[1] if bed_data is not None else None)

    args = (read_queue, processed_queue, remapper, norm_method, label, bed_data)
    for i in range(n_processors):
        p_args = args + (Path(output_path, f'{i + 1}.data.bin.tmp'), Path(output_path, f'{i + 1}.header.bin.tmp'))
        p = mp.Process(target=process_read, args=p_args, daemon=True)
        processors.append(p)

        p.start()

    return processors


def tqdm_with_time(msg, last_action_time):
    # Prints message with the difference between current time and last action time
    current_time = time.time()
    tqdm.write('>> ' + msg + f' {current_time - last_action_time}s')

    return current_time


def process_data(args: argparse.Namespace) -> None:
    start_time = time.time()
    last_action_time = start_time

    tqdm.write('>> Generating file list')
    file_queue = create_file_queue(args.input_path, args.recursive, args.n_producers)
    if file_queue is None:
        sys.exit('FAST5 file(s) not found.')

    info_args = get_info_args(args)
    last_action_time = tqdm_with_time(f'Parameters: {info_args}', last_action_time)

    if args.bed_path is None:
        bed_data = None
    else:
        last_action_time = tqdm_with_time('Extracting BED info', last_action_time)

        filter_method = bed_filter_factory(args.bed_filter)
        bed_data = extract_bed_positions(args.bed_path, filter_method)

    # Create output dir if it doesn't exist
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Writing extraction info
    info_path = Path(args.output_path, 'info.txt')
    BinaryWriter.write_extraction_info(info_path, info_args)

    last_action_time = tqdm_with_time('Building jobs', last_action_time)

    read_queue = mp.Queue()
    producers = start_producers(file_queue, read_queue, args.n_producers)

    processed_queue = None
    processors = worker_process_reads(read_queue, processed_queue, args.reference,
                                      args.norm_method, args.mapq, args.motif, args.index, args.window,
                                      args.label, args.del_method, bed_data, args.output_path, args.n_processors)

    tqdm_with_time('Extracting features', last_action_time)

    while True:
        if not any(p.is_alive() for p in producers):
            for _ in range(args.n_processors):
                read_queue.put(None)
        if not any(p.is_alive() for p in processors):
            break

    # Concatenate files
    BinaryWriter.on_extraction_finish(path=args.output_path)

    tqdm.write(f'Data generation finished. Total time: {time.time() - start_time}s')


def create_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input and output arguments
    parser.add_argument('input_path', type=Path,
                        help='Path to the input file or folder containing FAST5 files')

    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Flag to indicate if folder will be searched recursively (default: False)')

    parser.add_argument('output_path', type=Path,
                        help='Path to the desired output folder')

    # Other arguments
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to the reference file')

    parser.add_argument('--norm_method', type=str, default='standardization',
                        help='Function name to use for signal normalization (default: standardization)')

    parser.add_argument('--mapq', type=int, default=10,
                        help='Mapping quality threshold (default: 10)')

    parser.add_argument('--motif', type=str, default='CG',
                        help='Motif to be searched for in the sequences. Regular expressions can be used (default: CG)')

    parser.add_argument('--index', type=int, default=0,
                        help='Index of the central position in the motif (default: 0)')

    parser.add_argument('--window', type=int, default=8,
                        help='Window size around central position. Total k-mer size is: K = 2*W + 1 (default: 8)')

    parser.add_argument('--label', type=int, default=None,
                        help='Label to store for the given examples (default: None)')

    parser.add_argument('--del_method', type=str, default='concatenate_and_divide',
                        help='Deletion method to use for resolving deletions (default: concatenate_and_divide)')

    parser.add_argument('--n_producers', type=int, default=1,
                        help='Number of producers used for basecalling the input data (default: 1)')

    parser.add_argument('--n_processors', type=int, default=1,
                        help='Number of processors used for processing the basecalled data (default: 1)')

    # Bisulfite BED file arguments
    parser.add_argument('--bed_path', type=Path, default=None,
                        help='Path to BED file containing modification information (default: None)')

    parser.add_argument('--bed_filter', type=str, default=None,
                        help='BED filter method (e.g. high_confidence finds only high-confidence positions) (default: None)')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = create_arguments()

    process_data(arguments)