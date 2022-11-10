import abc
import enum
from collections import defaultdict
from glob import glob
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
from typing import Callable, Generic, List, NewType, Optional, Type, TypeVar, Union


T = TypeVar("T")



class Dependable:
    pass


class TimestampedDependable(Dependable):
    @abc.abstractmethod
    def timestamp(self) -> Optional[float]:
        pass


class AtomicDependable(Dependable):
    pass


class FileDependable(TimestampedDependable):
    def __init__(self, filename: str):
        super().__init__()
        assert type(filename) is str
        self.filename = filename

    def timestamp(self) -> Optional[float]:
        if not Path(self.filename).exists():
            return None
        return Path(self.filename).stat().st_mtime

    def __str__(self):
        return f"FileDependable(filename={self.filename})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, o):
        return self.filename == o.filename

    def __hash__(self):
        return hash(self.filename)


@dataclass(frozen=True)
class Dependency(Generic[T]):
    name: str
    dependable_type: Type[T]
    expand: Optional[Callable[[T], list[AtomicDependable]]]
    is_optional: bool = False


class Output(Dependable):
    pass


class FileOutput(Output, FileDependable):
    def __init__(self, filename: str):
        super().__init__(filename)



def _default_outcome_needs_update(
    output: Optional[Output],
    dependency: Optional[Dependency] = None,
    dependable: Optional[AtomicDependable] = None,
) -> bool:
    # Outcomes without outputs always trigger.
    if output is None:
        return True
    return output.needs_update(dependency, dependable)


@dataclass(frozen=True)
class Outcome(Generic[T]):
    name: str
    output_type: Optional[T]
    outputs_from_dependables: Callable[
        [dict[Dependency, list[AtomicDependable]]],
        dict[Output, dict[Dependency, list[AtomicDependable, bool]]]
,
    ]
    needs_update: Callable[[Optional[T], Optional[Dependency], Optional[AtomicDependable]], bool] = _default_outcome_needs_update
#    make_plan: Callable[[Optional[T], Optional[Dependency], Optional[AtomicDependable]], bool] = _default_outcome_make_plan

OutputMap = dict[Output, dict[Outcome, dict[Dependency, list[Dependable, bool]]]]


Executable = Callable[[], None]


DependenciesExpander = Callable[
    dict[Dependency, Dependable],
    dict[Dependency, list[AtomicDependable]],
]


#def _default_rule_make_plan(self, output_map: OutputMap, rule_args: T, outcome: Optional[Outcome], output: Optional[Output], has_outcomes: bool, has_dependencies: bool, has_outputs_needing_update: bool):
#    return has_outputs_needing_update


class Plan:
    def __init__(self, executable: Executable):
        self.executable = executable
        self.dependencies: list[Plan] = []
        self.needed_by: list[Plan] = []

    def add_dependency(self, plan: "Plan"):
        self.dependencies.append(plan)
        plan.needed_by.append(self)

    def execute(self, leaf_nodes):
        for l in leaf_nodes:
            l.executable()



@dataclass
class Rule(Generic[T]):
    name: str
    dependencies: Callable[[T], list[Dependency]]
    outcomes: Callable[[T], list[Outcome]]
    executable: Callable[[T], Executable]
    filter_for_needing_update: bool = True
    # needs_update = _default_rule_needs_update

    def artifact(
        self,
        name: str,
        args: T,
        dependencies_map: dict[Dependency, Dependable]
    )-> "Artifact":
        # TODO: Check that the keys of the map coincide with
        # the self.dependencies(args)
        return Artifact(name, self, args, dependencies_map)


OutputMap = dict[Output, dict[Outcome, dict[Dependency, list[Dependable, bool]]]]
 

class ArtifactOutputs(Dependable):
    def __init__(self, artifact: Dependable, outputs: set[Output]):
        self.artifact = artifact
        self.outputs = outputs


def expand_maybe_artifact_outputs(dependency, dependable):
    if isinstance(dependable, ArtifactOutputs):
        # TODO: can artifact outputs be expanded?
        return [ArtifactOutputs(dependable.artifact, [put]) for put in dependable.outputs]
    return dependency.expand(dependable)


class Artifact(Generic[T]):
    def __init__(
        self,
        name: str,
        rule: Rule,
        rule_args: T,
        dependencies_map: dict[Dependency, Dependable],
    ):
        self.name = name
        self.rule = rule
        self.rule_args = rule_args
        self.dependencies_map = dependencies_map
        self.expanded_dependencies_map = {k: expand_maybe_artifact_outputs(k, v) for k, v in self.dependencies_map.items()}
        print("Expanded dependencies map:", self.expanded_dependencies_map)
        self.outcomes = self.rule.outcomes(rule_args)
        self.output_map = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: set()
                    )
                )
            )
        self.output_info_to_artifact = defaultdict(lambda: None)
        self.has_outcomes = False
        self.dependencies = False
        self.has_outputs_needing_update = False
        for come in self.rule.outcomes(rule_args):
            self.has_outcomes = True
            for put, ency_to_ables in come.outputs_from_dependables(
                    self.expanded_dependencies_map
            ).items():
                for ency, ables in ency_to_ables.items():
                    self.has_dependencies = True
                    for able in ables:
                        if isinstance(able, ArtifactOutputs):
                            for able_output in able.outputs:
                                nu = come.needs_update(put, ency, able_output)
                                print("Dependable from artifact:", able_output, "nu:", nu)
                                self.has_outputs_needing_update = self.has_outputs_needing_update or nu
                                self.output_map[put][come][ency].add(
                                    (able_output, nu)
                                )
                            self.output_info_to_artifact[(put, come, ency)] = able.artifact
                        else:
                            nu = come.needs_update(put, ency, able)
                            self.has_outputs_needing_update = self.has_outputs_needing_update or nu
                            print("Dependable:", able, "nu:", nu)
                            self.output_map[put][come][ency].add(
                                (able, nu)
                            )


        # TODO: if an outcome doesn't have outputs, call needs_update with
        # output=None for each dependency/dependable (if there are no
        # dependencies/dependables, use None for those).
        self.executable = self.rule.executable(
            self.output_map
        )

    def outputs(self, outcome_name: str) -> set[AtomicDependable]:
        puts = set()
        for put in self.output_map:
           for come in self.output_map[put]:
               if come.name == outcome_name:
                   puts.add(put)
        return ArtifactOutputs(self, puts)

    def needs_update(self, outcome: Optional[Outcome] = None, output: Optional[Output] = None):
        return self.rule.needs_update(
            self.output_map,
            self.rule_args,
            outcome,
            output,
            self.has_outcomes,
            self.has_dependencies,
            self.has_outputs_needing_update
        )

    def make_plan(
        self,
        needed_by: Optional[list[Dependable]] = None,
        leaf_nodes: Optional[list[Dependable]] = None,
    ):
        artifact = self
        plan = Plan(artifact.executable)
        if needed_by is None:
            needed_by = []

        if leaf_nodes is None:
            leaf_nodes = []

        for n in needed_by:
            if n.name == artifact.name:
                raise ValueError(
                    "Circular dependency:\n"
                    + "\n->".join(n.name for n in needed_by)
                    + "\n->"
                    + artifact.name
                )

        needed_by.append(artifact)
        output_map = artifact.output_map
        output_info_to_artifact = artifact.output_info_to_artifact
        execute = False
        for put in output_map:
            print("Exploring output:", put)
            for come in output_map[put]:
                for ency in output_map[put][come]:
                    if (dep_artifact := output_info_to_artifact[(put,come,ency)]) is not None:
                        dep_plan, _ = dep_artifact.make_plan(needed_by, leaf_nodes)
                        if dep_plan is not None:
                            plan.add_dependency(dep_plan)
                            execute = True
                    for (able, nu) in output_map[put][come][ency]:
                        print("In make plan: able: ", able, "nu:", nu)
                        execute = execute or nu
        if not execute:
            return None, []
        if len(plan.dependencies) == 0:
            leaf_nodes.append(artifact)

        return plan, leaf_nodes


    def make(self):
        plan, leaf_nodes = self.make_plan()
        print("Leaf nodes:", leaf_nodes)
        plan.execute(leaf_nodes)



GlobExpr = NewType("GlobExpr", list[str])


class ObjsFromCsArgs(enum.Enum):
    fileName = 0
    fileNameList = 1
    globExpr = 2


GlobDependency = Dependency(
    name="glob",
    dependable_type=GlobExpr,
    expand=lambda globs: [FileDependable(f) for g in globs for f in glob(g)],
)


FileNameDependency = Dependency(
    name="file",
    dependable_type=str,
    expand=lambda x: [FileDependable(x)]
)


def expand_file_name_list(x: Union[list[str], ArtifactOutputs]):
    if isinstance(x, ArtifactOutputs):
        return ArtifactOutputs(x.artifact, [FileDependable(f.filename) for f in x.outputs])
    return [
        FileDependable(filename=f)
        for f in x
    ]


FileNameListDependency = Dependency(
    name="files",
    dependable_type=list[str],
    expand=expand_file_name_list,
)



def files_dependencies(args: ObjsFromCsArgs) -> list[Dependency]:
    if args is ObjsFromCsArgs.globExpr:
        return [GlobDependency]
    if args is ObjsFromCsArgs.fileName:
        return [FileDependency]
    if args is ObjsFromCsArgs.fileNameList:
        return [FileNameListDependency]


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


def timestamp_based_outcome(
    out: Output,
    ency: Dependency,
    able: AtomicDependable
) -> bool:
    print("out:", out)
    print("ency:", ency)
    print("able:", able)
    if out.timestamp() is None:
        return True
    if able.timestamp() is None:
        return True
    return able.timestamp() >= out.timestamp()


def objs_from_cs_outcomes(args: ObjsFromCsArgs) -> list[Outcome]:
    return [
        Outcome(
            name="obj_files",
            output_type=list[FileOutput],
            outputs_from_dependables=obj_files_outputs_from_dependables,
            needs_update=timestamp_based_outcome,
        )
    ]


def _do_execute_in_shell(command: list[str]):
    print("Executing:", command)
    subprocess.check_output(command)

def shell_executable(command: list[str]) -> Executable:
    return lambda: _do_execute_in_shell(command)


def dependables_needing_update(output_map: OutputMap) -> list[Dependable]:
    return [
        ables_and_needs[0]
        for put in output_map
        for come in output_map[put]
        for ency in output_map[put][come]
        for ables_and_needs in output_map[put][come][ency]
        if ables_and_needs[1]
    ]



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


def this_dir_objs():
    return objs_from_cs.artifact(
        "this_dir_objs",
        ObjsFromCsArgs.globExpr,
        {GlobDependency: ["*.c"]}
    )


# if this_dir_objs.needs_update():
#     this_dir_objs.executable()
# print("Outputs", this_dir_objs.output_map.keys())


def single_file_output_from_dependables(filename: str, deps_to_ables: dict[Dependency, list[AtomicDependable]]):
    return {
        FileOutput(filename): {
            ency: set([able for able in deps_to_ables[ency]])
            for ency in deps_to_ables
        }
    }



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


exec_from_objs = Rule(
    name = "exec_from_objs",
    dependencies = files_dependencies,
    outcomes = exec_from_objs_outcomes,
    executable = exec_from_objs_executable,
)


def this_dir_exec():
    return exec_from_objs.artifact(
        "this_dir_exec",
        ExecFromObjsArgs(exec_name="ppp"),
        {FileNameListDependency: this_dir_objs().outputs("obj_files")}
    )


# if this_dir_objs.needs_update():
#     this_dir_objs.executable()
# print("Outputs", this_dir_objs.output_map.keys())

# this_dir_exec.make()

