from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from weather.local_capacity import commit_free_bytes,local_training_lease


class LocalCapacityChecks(unittest.TestCase):
    def test_commit_headroom_is_readable_or_explicitly_unavailable(self):
        value=commit_free_bytes()
        self.assertTrue(value is None or isinstance(value,int) and value>=0)

    def test_second_process_waits_and_lease_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lease.lock'
            script=('import sys;from weather.local_capacity import local_training_lease\n'
                'try:\n with local_training_lease(sys.argv[1],poll_seconds=.02,timeout_seconds=.15):pass\n'
                'except TimeoutError:sys.exit(0)\n'
                'sys.exit(7)\n')
            with local_training_lease(path):
                completed=subprocess.run([sys.executable,'-B','-c',script,str(path)],capture_output=True,text=True,timeout=5)
                self.assertEqual(completed.returncode,0,completed.stderr)
            with local_training_lease(path,timeout_seconds=.2):pass


if __name__=='__main__':unittest.main()
