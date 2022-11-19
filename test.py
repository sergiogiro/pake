from pathlib import Path
import unittest
from unittest.mock import call, patch

import pake as pk


executed_command = []

def mock_shell_executable(command):
    def execute():
        executed_command[:] = command
    return execute


class TestThisDirExec(unittest.TestCase):
    @patch("pake._do_execute_in_shell")
    def test_this_dir_exec(self, mock_execute_in_shell):
        Path("a.o").touch()
        Path("b.o").unlink(missing_ok=True)
        pk.this_dir_exec.make()
   
        print("Call args list:", mock_execute_in_shell.call_args_list)

        self.assertEqual(mock_execute_in_shell.call_args_list, [call(["gcc", "-c", "b.c"]), call(["gcc", "-o", "ppp", "a.o", "b.o"])])
    

if __name__ == "__main__":
    unittest.main()

