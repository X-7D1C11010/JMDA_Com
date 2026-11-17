"""A fixed target label budget; unknown instances never enter class alignment."""
from collections import defaultdict, Counter
import hashlib
import json
import random
from pathlib import Path

from weather.class_matching import protocol_metadata, target_truth_for_training


class TargetLabelBudget:
    def __init__(self, paths, labels, per_class, seed):
        if per_class<=0 or len(paths)!=len(labels) or len(paths)!=len(set(paths)):
            raise ValueError('Invalid fixed target label budget')
        groups=defaultdict(list)
        for path,label in zip(paths,labels):groups[int(label)].append(path)
        self.target_lookup={}
        for label, candidates in sorted(groups.items()):
            if len(candidates)<per_class:raise ValueError(f'Target class {label} has fewer than {per_class} samples')
            candidates=sorted(candidates)
            random.Random(seed+100003*label).shuffle(candidates)
            self.target_lookup.update({path:label for path in candidates[:per_class]})
        self.per_class=per_class;self.seed=seed;self.total=len(paths)
        self.steps=None;self.source_samples=None;self.batch_sizes=None

    @classmethod
    def from_dataset(cls,target,label_map,per_class,seed):
        # Full annotations are used once to construct a stratified label budget.
        # Only selected instance labels are retained by the trainer.
        return cls(target.paths,target_truth_for_training(target,label_map),per_class,seed)

    def labels_for(self,paths):
        return [self.target_lookup.get(path,-1) for path in paths]

    def metadata(self,weight):
        selected=sorted(self.target_lookup.items())
        portable=sorted((label,Path(path).name) for path,label in selected)
        if len(set(portable))!=len(portable):raise ValueError('Ambiguous class/filename identifiers in labelled target pool')
        return {**protocol_metadata(True,weight,self.per_class),
            'rule':'independent full-coverage source/target batches; class alignment only for selected labelled targets',
            'labels_per_class':self.per_class,'label_selection_seed':self.seed,
            'labelled_target_samples':len(selected),'unlabelled_target_samples':self.total-len(selected),
            'labelled_target_class_counts':dict(Counter(self.target_lookup.values())),
            'selected_labelled_samples':[{'path':p,'label':y} for p,y in selected],
            'label_pool_sha256':hashlib.sha256(json.dumps(selected,ensure_ascii=False).encode()).hexdigest(),
            'label_pool_portable_sha256':hashlib.sha256(json.dumps(portable,ensure_ascii=False).encode()).hexdigest(),
            'stratification_uses_annotations_in_split_preparation_only':True,
            'unselected_truth_used_for_pairing_or_alignment':False,
            'unlabelled_targets_used_by_native_DA':True,'native_batches_true_class_matched':False,
            'optimizer_steps_per_epoch':self.steps,
            'source_exposures_per_epoch':self.steps*self.batch_sizes[0],
            'target_exposures_per_epoch':self.steps*self.batch_sizes[1],
            'all_source_and_target_samples_covered_each_epoch':True,'matched_alignment_weight':weight}
