from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, Sampler

from lgmps.data import AssembledSequence, IndexedJsonlDataset, SequenceAssembler, pack_sequences
from lgmps.data.assembler import PackedBatch


class TrainingSequenceDataset(Dataset[AssembledSequence]):
    def __init__(self, records: IndexedJsonlDataset, assembler: SequenceAssembler) -> None:
        self.records = records
        self.assembler = assembler

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AssembledSequence:
        return self.assembler.assemble(self.records[index], mode="training")


class EpochRandomSampler(Sampler[int]):
    """Epoch-addressable permutation sampler that makes mid-epoch resume exact."""

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def collate_sequences(sequences: list[AssembledSequence]) -> PackedBatch:
    return pack_sequences(sequences)
