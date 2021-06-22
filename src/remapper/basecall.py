from queue import Empty
from ont_fast5_api.conversion_tools.conversion_utils import get_fast5_file_list
from ont_fast5_api.fast5_interface import get_fast5_file
from pyguppy_client_lib.pyclient import PyGuppyClient

import numpy as np
from dataclasses import dataclass
import random
import argparse
import multiprocessing as mp
import time

from typing import Optional, Tuple, List, Iterator, Dict, Type, TypeVar, Any, Union, Set

from .util import Interval


@dataclass(frozen=True)
class GuppyRead:
    read_id: str
    raw_data: np.ndarray
    daq_scaling: float
    daq_offset: float
    read_tag: Optional[int] = None


U = TypeVar('U', bound='BasecallData')


@dataclass(frozen=True)
class BasecallData:
    read: GuppyRead
    seq: str
    move: np.ndarray
    trimmed_samples: int
    model_stride: int

    @classmethod
    def from_called_dict(cls: Type[U], called_dict: Dict[str, Any]) -> U:
        metadata = called_dict['metadata']
        datasets = called_dict['datasets']

        read_id = metadata['read_id']
        raw_data = datasets['raw_data']
        scaling = metadata['daq_scaling']
        offset = metadata['daq_offset']
        read = GuppyRead(read_id, raw_data, scaling, offset)

        seq = datasets['sequence']
        move = datasets['movement']
        trimmed_samples = metadata['trimmed_samples']
        model_stride = metadata['model_stride']

        return cls(read, seq, move, trimmed_samples, model_stride)


def process_completed_reads(client: PyGuppyClient, sent_reads: Set[str],
                            read_queue: mp.Queue) -> None:
    completed_reads = client.get_completed_reads()

    for completed_read in completed_reads:
        data = BasecallData.from_called_dict(completed_read)
        read_queue.put(data)
        sent_reads.remove(data.read.read_id)


def basecall_files(file_queue: mp.Queue, read_queue: mp.Queue) -> None:
    with PyGuppyClient("127.0.0.1:5555", "dna_r9.4.1_450bps_hac", move_and_trace_enabled=True) as client:
        sent_reads = set()

        while True:
            file = file_queue.get()
            if file is None:
                break

            for read in get_reads(file):
                client.pass_read(read.__dict__)
                sent_reads.add(read.read_id)

                while len(sent_reads) >= 10:
                    process_completed_reads(client, sent_reads, read_queue)
                time.sleep(0.01)

        while len(sent_reads) > 0:
            process_completed_reads(client, sent_reads, read_queue)

        print('Done')


def sequence_to_raw(data: BasecallData) -> Tuple[List[Interval], int]:
    raw_data = data.read.raw_data

    seq_to_move_idx = np.nonzero(data.move)[0]

    raw_start_idx = data.trimmed_samples
    seq_to_raw_start = raw_start_idx + seq_to_move_idx * data.model_stride

    seq_to_raw_interval = [Interval(s, e) for s, e in zip(seq_to_raw_start[:-1], seq_to_raw_start[1:])]
    seq_to_raw_interval.append(Interval(seq_to_raw_start[-1], len(raw_data)))  # Adding the last event

    return seq_to_raw_interval, raw_start_idx


def get_reads(path: str) -> Iterator[GuppyRead]:
    with get_fast5_file(path, 'r') as f:
        for read in f.get_reads():
            signal = read.get_raw_data()

            channel_info = read.get_channel_info()
            scaling = float(channel_info['range']) / float(channel_info['digitisation'])
            offset = float(channel_info['offset'])

            random_tag = random.randrange(2 ** 32)

            yield GuppyRead(read.read_id, signal, scaling, offset, read_tag=random_tag)


def create_file_queue(path: str, recursive: bool, n_producers: int) -> Optional[mp.Queue]:
    files = get_fast5_file_list(path, recursive)
    if len(files) == 0:
        return None
    # files.insert(10_000, None)  # TODO for testing

    file_queue = mp.Queue()
    for file in files:
        file_queue.put(file)
    for _ in range(n_producers):
        file_queue.put(None)

    return file_queue


def start_producers(file_queue: mp.Queue, read_queue: mp.Queue, n_producers: int) -> List[mp.Process]:
    producers = []

    for _ in range(n_producers):
        producer = mp.Process(target=basecall_files, args=(file_queue, read_queue), daemon=True)
        producers.append(producer)

        producer.start()

    return producers


def create_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('--n_producers', type=int, default=1)

    return parser.parse_args()


def main():
    import sys

    args = create_arguments()

    read_queue = mp.Queue()
    file_queue = create_file_queue(args.input_path, args.recursive, args.n_producers)

    print('Starting with basecalling')

    producers = start_producers(file_queue, read_queue, args.n_producers)

    reads = 0
    while True:
        all_finished = not any([p.is_alive() for p in producers])
        if all_finished:
            break

        try:
            read = read_queue.get(timeout=10)
            reads += 1

            if reads % 100 == 0:
                print(reads)
        except Empty:
            pass


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)
