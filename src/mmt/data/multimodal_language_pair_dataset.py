import logging
from functools import lru_cache
import os

import numpy as np
import torch
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    data_utils
    )

logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def merge_feats(key):
        """Convert a list of 1d tensors into a 2d tensor.
        Assumes all feats are the same length.
        """
        values = [s[key] for s in samples]
        size = values[0].size()
        batch_size = len(values)
        res = values[0].new(batch_size, *size).fill_(0)
        for i, v in enumerate(values):
            res[i,:] = v
        return res

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    feats = merge_feats("feats")
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    feats = feats.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "feats": feats
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch


    
class MultimodalLanguagePairDataset(LanguagePairDataset):
    """
    A triplet of torch.utils.data.Datasets.

    Args:
        feats (torch.utils.data.Dataset): image/video features dataset to wrap
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """
    
    def __init__(
        self,
        src,            
        src_sizes,
        src_dict,
        feats=None,
        feats_sizes=None,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1):
        
        super().__init__(
            src,            
            src_sizes,
            src_dict,
            tgt,
            tgt_sizes,
            tgt_dict,
            left_pad_source,
            left_pad_target,
            shuffle,
            input_feeding,
            remove_eos_from_source,
            append_eos_to_target,
            align_dataset,
            constraints,
            append_bos,
            eos,
            num_buckets,
            src_lang_id,
            tgt_lang_id,
            pad_to_multiple)

        if feats is not None:
            assert len(src) == len(
                feats
            ), "Source, target, and multimodal features must contain the same number of examples"
        self.feats = feats
        self.feats_sizes = feats_sizes

    def __getitem__(self, index):
        example = super().__getitem__(index)
        # example = {
        #     "id": index,
        #     "source": src_item,
        #     "target": tgt_item,
        #     "alignment": ..optional
        #     "constraints": ...optional
        # }        
        if self.feats is not None:
            example["feats"] = self.feats[index]
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `feats` (FloatTensor): image features of shape `(bsz, feat_len)`.
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

#    @property
#    def supports_prefetch(self):
#        return getattr(self.src, "supports_prefetch", False) and (
#            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
#        ) and (
#            getattr(self.feats, "supports_prefetch", False) or self.feats is None
#        )
#
#    def prefetch(self, indices):
#        self.src.prefetch(indices)
#        if self.tgt is not None:
#            self.tgt.prefetch(indices)
#        if self.align_dataset is not None:
#            self.align_dataset.prefetch(indices)
#        if self.feats is not None:
#            self.feats.prefetch(indices)
#


class NpzFolderDataset(torch.utils.data.Dataset):
    def __init__(self, features_dir, file_list, file_prefix="", file_suffix=""):
        self.features_dir = features_dir
        self.image_ids = np.array(list(map(
            lambda x: os.path.splitext(x)[0],
            np.loadtxt(file_list + ".txt", dtype=str))))
        self.prefix = file_prefix
        self.suffix = file_suffix

    def __getitem__(self, index):
        npz_file = os.path.join(
            self.features_dir,
            self.prefix + self.image_ids[index] + self.suffix)
        d = np.load(npz_file)
        x = torch.from_numpy(d["x"])
        return x

    def __len__(self):
        return len(self.image_ids)


class NumpyMmapDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, dtype=np.float32):
        self.data = np.load(data_path, mmap_mode="r")
        self.transform = transform
        self.dtype = dtype

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        x = self.data[index].astype(self.dtype)
        if self.transform is not None:
            x = self.transform(np.expand_dims(x, axis=0)).squeeze(axis=0)
        x = torch.from_numpy(x)

        return x

    def __len__(self):
        return len(self.data)

            
class NumpyMmapDatasetCached(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, dtype=np.float32):
        self.data = np.load(data_path, mmap_mode="r")
        self.transform = transform
        self.dtype = dtype
        self.cache = None
        self.cache_index = {}  

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        if index in self.cache_index:
            x = self.cache[self.cache_index[index]]
        else:
            x = self.data[index].astype(self.dtype)
            if self.transform is not None:
                x = self.transform(np.expand_dims(x, axis=0)).squeeze(axis=0)
            x = torch.from_numpy(x)

        return x

    def __len__(self):
        return len(self.data)

    @property
    def supports_prefetch(self):
        return True

    
    def prefetch(self, indices):
        x = self.data[indices].astype(self.dtype)

        if self.transform is not None:
            x = self.transform(x)

        self.cache = torch.from_numpy(x)
        self.cache_index.clear()
        for (i,ix) in enumerate(indices):
            self.cache_index[ix] = i
