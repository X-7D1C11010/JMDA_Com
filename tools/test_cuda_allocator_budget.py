import unittest
from unittest.mock import patch
import torch
from weather.train_vds_core import configure_allocator_budget


class DeviceIndexChecks(unittest.TestCase):
    def test_implicit_cuda_resolves_current_index_and_explicit_cuda_keeps_it(self):
        with patch('torch.cuda.current_device',return_value=0) as current, patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_allocator_budget(torch.device('cuda'),.16)
            setter.assert_called_once_with(.16,0);current.assert_called_once()
        with patch('torch.cuda.current_device') as current,patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_allocator_budget(torch.device('cuda:1'),.16)
            setter.assert_called_once_with(.16,1);current.assert_not_called()

    def test_legacy_default_and_cpu_never_initialize_cuda(self):
        with patch('torch.cuda.current_device') as current,patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_allocator_budget(torch.device('cuda'),1.)
            configure_allocator_budget(torch.device('cpu'),.16)
            current.assert_not_called();setter.assert_not_called()


if __name__=='__main__':unittest.main()
