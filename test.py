from pathlib import Path
import unittest
from unittest.mock import call, patch

from pake import *


executed_command = []


def obj_files_outputs_from_dependables(deps_to_ables: dict[Dependency, list[AtomicDependable]]):
    print("Deps to ables:", deps_to_ables)
    def extension_to_o(path):
        spath = path.rsplit(".", 1)
        if len(spath) < 2:
            return None
            # raise ValueError(f"Path doesn't have an extension: [{spath}]")
        p, e = spath
        if os.sep in e:
            return None
            # raise ValueError(
            #     "Path doesn't have an extension (only dot is at the left "
            #     f"of a {os.sep}): [{spath}]"
            # )
        return p + ".o"

    # TODO: consider output files tied to multiple dependencies/ables?
    ret = {
        FileOutput(extension_to_o(able.filename)): { ency: (able,) }
        for ency in deps_to_ables
        for able in deps_to_ables[ency]
        if extension_to_o(able.filename) is not None
    }
    print("Obj files outputs:", ret)
    return ret


def objs_from_cs_outcomes(args: ObjsFromCsArgs) -> list[Outcome]:
    return [
        Outcome(
            name="obj_files",
            output_type=list[FileOutput],
            outputs_from_dependables=obj_files_outputs_from_dependables,
            needs_update=timestamp_based_outcome,
        )
    ]


@dataclass
class ExecFromObjsArgs:
    exec_name: str


def exec_from_objs_outcomes(args: ExecFromObjsArgs) -> list[Outcome]:
    return [
        Outcome(
            name="exec",
            output_type=FileOutput,
            outputs_from_dependables=lambda deps_to_ables: single_file_output_from_dependables(args.exec_name, deps_to_ables),
            needs_update=timestamp_based_outcome,
        )
    ]


def exec_from_objs_executable(output_map: OutputMap):
    output = list(output_map.keys())[0]
    outcome = list(output_map[output].keys())[0]
    print("Output map:", output_map)
    ables = [
        ables_and_needs[0]
        for put in output_map
        for come in output_map[put]
        for ency in output_map[put][come]
        for ables_and_needs in output_map[put][come][ency]
    ]
    print("Ables: ", ables)
    return shell_executable(
        ["gcc", "-o", output.filename]  + sorted([able.filename for able in ables])
    )


def objs_from_cs_executable(output_map: OutputMap):
    print("Output map:", output_map)

    return shell_executable(
        ["gcc", "-c"] + sorted([
            able.filename
            for able in dependables_needing_update(output_map)
        ])
    )


objs_from_cs = Rule(
    name = "objs_from_cs",
    dependencies = files_dependencies,
    outcomes = objs_from_cs_outcomes,
    executable = objs_from_cs_executable,
)


this_dir_objs = objs_from_cs.artifact(
    "this_dir_objs",
    ObjsFromCsArgs.globExpr,
    {GlobDependency: ["*.c"]}
)


exec_from_objs = Rule(
    name = "exec_from_objs",
    dependencies = files_dependencies,
    outcomes = exec_from_objs_outcomes,
    executable = exec_from_objs_executable,
)


this_dir_exec = exec_from_objs.artifact(
    "this_dir_exec",
    ExecFromObjsArgs(exec_name="ppp"),
    {FileNameListDependency: this_dir_objs.outputs("obj_files")}
)


def mock_shell_executable(command):
    def execute():
        executed_command[:] = command
    return execute


class TestThisDirExec(unittest.TestCase):
    @patch("pake._do_execute_in_shell")
    def test_this_dir_exec(self, mock_execute_in_shell):
        Path("a.o").touch()
        Path("b.o").unlink(missing_ok=True)
        this_dir_exec.make()
   
        print("Call args list:", mock_execute_in_shell.call_args_list)

        self.assertEqual(mock_execute_in_shell.call_args_list, [call(["gcc", "-c", "b.c"]), call(["gcc", "-o", "ppp", "a.o", "b.o"])])
    

if __name__ == "__main__":
    unittest.main()

