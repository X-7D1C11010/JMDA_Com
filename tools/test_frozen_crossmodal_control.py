"""A frozen-encoder control must retain trainable source projection losses."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tools import test_crossmodal_loss as helpers
from tools.test_weather_vds_core import fixture
from weather import train_vds_core as weather


class FrozenCrossmodalControlChecks(unittest.TestCase):
    def test_actual_entry_can_train_projection_with_frozen_encoders_and_keeps_native_DA_off(self):
        torch.set_num_threads(2)
        helper=helpers.CrossmodalChecks()
        def model(method,*unused,**kwargs):
            result=helper.make_model(method,kwargs['num_classes'])
            result.rgb_encoder.requires_grad_(False);result.nir_encoder.requires_grad_(False)
            result.encoder_training_scope='frozen';result.freeze_image_encoders=True
            return result
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained','--trade-off','0',
                '--source-crossmodal-weight','.1','--crossmodal-projection-dim','8','--encoder-training-scope','frozen'])
            with patch.object(weather,'AdaptationModel',model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            c=json.loads((output/'run_config.json').read_text())
            self.assertEqual(sum(g['parameters'] for g in c['optimizer_groups'] if g['role'].startswith('pretrained')),0)
            update=json.loads((output/'updates.jsonl').read_text().splitlines()[0])
            self.assertGreater(update['weighted_source_crossmodal'],0)
            self.assertFalse(update['core_adaptation_graph_in_loss'])
            grad=json.loads((output/'gradients.jsonl').read_text().splitlines()[0])
            projection=grad['enhancement_components']['source_crossmodal']['crossmodal_projection']
            self.assertGreater(projection['weighted_DA_norm'],0)
            for branch in ['VIS','IR']:
                self.assertEqual(grad['gradients'][branch]['CE_norm'],0)
            args.crossmodal_projection_dim=0
            args.save_dir=str(root/'no_projection')
            with patch.object(weather,'AdaptationModel',model),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'trainable contrastive projection'):weather.run(args)


if __name__=='__main__':unittest.main()
