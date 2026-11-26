from __future__ import annotations

import argparse
import json

import torch

from lgmps.configuration import PUBLIC_DATASETS, load_experiment
from lgmps.data import IndexedJsonlDataset, SequenceAssembler, pack_sequences
from lgmps.inference.beam_search import generate_semantic_ids
from lgmps.inference.evaluation import load_inference_model, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic document IDs for one local record.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=PUBLIC_DATASETS)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--record-index", type=int, default=0)
    arguments = parser.parse_args()
    config = load_experiment(arguments.config, dataset=arguments.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(config, arguments.checkpoint)
    model = load_inference_model(config, checkpoint, device)
    records = IndexedJsonlDataset(
        config.data.train_file,
        config.data.index_file,
        dataset_name=config.dataset,
    )
    record = records[arguments.record_index]
    assembler = SequenceAssembler(
        query_term_slots=config.data.query_term_slots,
        add_query_markers=config.data.add_query_markers,
        add_item_marker=config.data.add_item_marker,
        maximum_tokens=config.data.maximum_tokens,
        positive_event_mask=config.data.positive_event_mask,
        target_policy=config.data.target_policy,
    )
    sequence = assembler.assemble(record, mode="test_prefix")
    result = generate_semantic_ids(
        model,
        pack_sequences([sequence]).to(device),
        beam_size=config.inference.beam_size,
        return_sequences=config.inference.return_sequences,
        temperature=config.inference.temperature,
    )
    print(
        json.dumps(
            {
                "record_id": record.record_id,
                "user_id": record.user_id,
                "predicted_semantic_ids": result.semantic_ids,
                "beam_scores": result.scores,
                "target_semantic_ids": sequence.target_semantic_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
