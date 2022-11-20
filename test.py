import os

import unittest
from unittest.mock import call, patch

from pake import *
from dataclasses import dataclass

executed_command = []


def obj_files_outputs_from_dependables(deps_to_ables: dict[Dependency, list[FileDependable]]):
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
        FileOutput(extension_to_o(able.filename)): {ency: (able,)}
        for ency in deps_to_ables
        for able in deps_to_ables[ency]
        if extension_to_o(able.filename) is not None
    }
    print("Obj files outputs:", ret)
    return ret


@dataclass
class ExecFromObjsArgs(object):
    exec_name: str


def exec_from_objs_executable(output_map: OutputMap):
    om = output_map()
    output = list(om.keys())[0]
    ables = dependables_needing_update(output_map)
    print("Ables: ", ables)
    return shell_executable(
        ["gcc", "-o", output.filename] + sorted([able.filename for able in ables])
    )


def objs_from_cs_executable(output_map: OutputMap[any, any, any, FileDependable]) -> Action:
    print("Output map:", output_map)

    return shell_executable(
        ["gcc", "-c"] + sorted([
            able.filename
            for able in dependables_needing_update(output_map)
        ])
    )


class TestThisDirExec(unittest.TestCase):
    @patch("pake._do_execute_in_shell")
    def test_this_dir_exec(self, mock_execute_in_shell):
        class ExecFromObjs(
            Rule[ExecFromObjsArgs],
            dependencies=files_dependencies,
            executable=Executable(exec_from_objs_executable),
        ):
            exec = Outcome.from_args(
                lambda args:
                    Outcome(
                        name="exec",
                        output_type=FileOutput,
                        outputs_from_dependables=lambda deps_to_ables: single_file_output_from_dependables(
                            args.exec_name, deps_to_ables
                        ),
                        needs_update=timestamp_based_outcome,
                    )
            )

        class ObjsFromCs(
            Rule,
            dependencies=files_dependencies,
            executable=Executable(objs_from_cs_executable),
        ):
            obj_files = Outcome(
                name="obj_files",
                output_type=list[FileOutput],
                outputs_from_dependables=obj_files_outputs_from_dependables,
                needs_update=timestamp_based_outcome,
            )

        this_dir_objs = artifact(
            ObjsFromCs,
            "this_dir_objs",
            ObjsFromCsArgs.globExpr,
            {**file_globs_dependency(["*.c"])}
        )

        this_dir_exec = artifact(
            ExecFromObjs,
            "this_dir_exec",
            ExecFromObjsArgs(exec_name="ppp"),
            {FileNameListDependency: this_dir_objs.outputs("obj_files")}
        )

        Path("a.o").touch()
        Path("b.o").unlink(missing_ok=True)
        this_dir_exec.make()
        print("Call args list:", mock_execute_in_shell.call_args_list)

        self.assertEqual(mock_execute_in_shell.call_args_list, [call(["gcc", "-c", "b.c"]), call(["gcc", "-o", "ppp", "a.o", "b.o"])])


if __name__ == "__main__":
    unittest.main()
