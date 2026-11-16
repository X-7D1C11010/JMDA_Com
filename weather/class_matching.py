"""Explicit target-truth-guided sampling; this protocol is NOT strict UDA.

Target labels are read from target_train metadata only and kept in the sampler,
not passed as a target class tensor to the model. Optional matched feature
alignment is separately declared as target-label-guided supervision. Each source/target batch position has the
same true class. Both datasets are fully covered, with cyclic within-class
oversampling instead of a Cartesian product of every cross-domain pair.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import random


PROTOCOL = "LABEL_GUIDED_CLASS_MATCHING"
ALIGNMENT_PROTOCOL = "LABEL_GUIDED_CLASS_ALIGNMENT"


def protocol_metadata(enabled,alignment_weight=0.):
    if alignment_weight and not enabled:raise ValueError('True-class alignment requires label-guided matching')
    return {"protocol":(ALIGNMENT_PROTOCOL if alignment_weight else PROTOCOL) if enabled else "UDA",
            "target_training_labels_used":bool(enabled),
            "target_true_labels_used_for_sampling":bool(enabled),
            "target_true_labels_used_in_classification_loss":False,
            "target_true_labels_used_in_alignment_loss":bool(alignment_weight),
            "strict_unsupervised":not enabled}


def target_truth_for_training(target, label_map):
    from tools.uda_data import read_image_list
    from tools.data_paths import remap_data_path
    path=Path(target.dataset.label_list_path)
    if not path.name.endswith('_train.txt'):
        raise ValueError('Class matching may read target_train metadata only')
    by_path={}
    for raw,label in read_image_list(path):
        canonical=str(Path(remap_data_path(raw)).resolve())
        if canonical in by_path:raise ValueError('Duplicate target training path')
        if label not in label_map:raise ValueError('Target class lies outside the shared training support')
        by_path[canonical]=label_map[label]
    actual=[str(Path(p).resolve()) for p in target.paths]
    if set(actual)!=set(by_path) or len(actual)!=len(by_path):
        raise ValueError('Target class metadata and actual training pairs differ')
    return [by_path[p] for p in actual]


class MatchedClassPlan:
    def __init__(self,source_labels,target_labels,source_paths,target_paths,batch_size,seed):
        if batch_size<2:raise ValueError('Matched batches must contain at least two samples')
        self.source_labels=list(map(int,source_labels));self.target_labels=list(map(int,target_labels))
        if not self.source_labels or not self.target_labels:raise ValueError('Both training domains must contain samples')
        if set(self.source_labels)!=set(self.target_labels):raise ValueError('Source/target class supports must be exactly equal')
        if len(source_paths)!=len(source_labels) or len(target_paths)!=len(target_labels):raise ValueError('Path/label size mismatch')
        self.source_lookup=dict(zip(source_paths,self.source_labels));self.target_lookup=dict(zip(target_paths,self.target_labels))
        if len(self.source_lookup)!=len(source_paths) or len(self.target_lookup)!=len(target_paths):raise ValueError('Duplicate training paths')
        self.batch_size,self.seed=batch_size,seed
        self.classes=sorted(set(self.source_labels))
        self.source_counts=Counter(self.source_labels);self.target_counts=Counter(self.target_labels)
        self.pairs_per_epoch=sum(max(self.source_counts[c],self.target_counts[c]) for c in self.classes)
        self.steps=math.ceil(self.pairs_per_epoch/batch_size)
        self._epoch=None;self._batches=None

    def batches(self,epoch):
        if self._epoch==epoch:return self._batches
        rng=random.Random(self.seed+100003*int(epoch))
        groups=[]
        for labels in [self.source_labels,self.target_labels]:
            table=defaultdict(list)
            for index,label in enumerate(labels):table[label].append(index)
            groups.append(table)
        pairs=[]
        for label in self.classes:
            count=max(self.source_counts[label],self.target_counts[label])
            sides=[]
            for table in groups:
                indices=[]
                while len(indices)<count:
                    cycle=table[label].copy();rng.shuffle(cycle);indices.extend(cycle)
                sides.append(indices[:count])
            pairs.extend(zip(*sides))
        rng.shuffle(pairs)
        # Pad with synchronized pairs, preserving class correspondence.
        required=self.steps*self.batch_size
        pairs += [pairs[i%len(pairs)] for i in range(required-len(pairs))]
        self._batches=[pairs[i:i+self.batch_size] for i in range(0,required,self.batch_size)]
        self._epoch=epoch
        return self._batches

    def assert_batch(self,source_paths,target_paths,source_labels):
        a=[self.source_lookup[p] for p in source_paths]
        b=[self.target_lookup[p] for p in target_paths]
        if a!=b or a!=list(map(int,source_labels)):
            raise RuntimeError('Label-guided source/target batch is not class-matched at every position')
        return b

    def metadata(self,alignment_weight=0.):
        exposures=self.steps*self.batch_size
        return {**protocol_metadata(True,alignment_weight),'rule':'same true class at every source/target batch position',
                'class_ids':self.classes,'source_class_counts':dict(self.source_counts),
                'target_class_counts':dict(self.target_counts),'optimizer_steps_per_epoch':self.steps,
                'source_exposures_per_epoch':exposures,'target_exposures_per_epoch':exposures,
                'source_repeated_exposures_per_epoch':exposures-len(self.source_labels),
                'target_repeated_exposures_per_epoch':exposures-len(self.target_labels),
                'all_source_and_target_samples_covered_each_epoch':True,
                'pair_rule':'max(source_count,target_count) per class, then synchronized batch padding',
                'cartesian_cross_domain_pair_enumeration':False,'target_label_source':'target_train metadata only',
                'target_labels_enter_model_objective':bool(alignment_weight),
                'matched_alignment_weight':alignment_weight}


class MatchedClassBatchSampler:
    def __init__(self,plan,domain):self.plan,self.domain,self.epoch=plan,domain,0
    def __len__(self):return self.plan.steps
    def __iter__(self):
        for pairs in self.plan.batches(self.epoch):yield [p[self.domain] for p in pairs]


def make_matched_samplers(source,target,source_batch_size,target_batch_size,label_map,seed):
    target_batch_size=target_batch_size if target_batch_size is not None else source_batch_size
    if source_batch_size!=target_batch_size:
        raise ValueError('True-class position matching requires equal source/target batch sizes; set both explicitly')
    labels=[label for _,label in source.dataset.samples]
    plan=MatchedClassPlan(labels,target_truth_for_training(target,label_map),source.paths,target.paths,source_batch_size,seed)
    return plan,[MatchedClassBatchSampler(plan,i) for i in [0,1]]
