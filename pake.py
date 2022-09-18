import abc
import enum
from collections import defaultdict
from glob import glob
from dataclasses import dataclass
import os
import subprocess
from typing import Callable, Generic, List, NewType, Optional, Type, TypeVar, Union


T = TypeVar("T")


@dataclass(frozen=True)
class Dependable:
    pass


@dataclass(frozen=True)
class AtomicDependable(Dependable):
    pass


@dataclass(frozen=True)
class FileDependable(AtomicDependable):
    filename: str


@dataclass(frozen=True)
class Dependency(Generic[T]):
    name: str
    dependable_type: Type[T]
    expand: Optional[Callable[[T], list[AtomicDependable]]]
    is_optional: bool = False


@dataclass(frozen=True)
class Output(Dependable):
    pass


@dataclass(frozen=True)
class FileOutput(Output):
    filename: str



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


OutputMap = dict[Output, dict[Outcome, dict[Dependency, list[Dependable, bool]]]]


Executable = Callable[[], None]


DependenciesExpander = Callable[
    dict[Dependency, Dependable],
    dict[Dependency, list[AtomicDependable]],
]


def _default_rule_needs_update(self, output_map: OutputMap, rule_args: T, outcome: Optional[Outcome], output: Optional[Output], has_outcomes: bool, has_dependencies: bool, has_outputs_needing_update: bool):
    # Rules without outcomes or dependencies always trigger.
    return (not has_outcomes) or (not has_dependencies) or has_outputs_needing_update


@dataclass
class Rule(Generic[T]):
    name: str
    dependencies: Callable[[T], list[Dependency]]
    outcomes: Callable[[T], list[Outcome]]
    executable: Callable[[T], Executable]
    filter_for_needing_update: bool = False
    needs_update = _default_rule_needs_update

    def artifact(
        self,
        name: str,
        args: T,
        dependencies_map: dict[Dependency, Dependable]
    )-> "Artifact":
        # TODO: Check that the keys of the map coincide with
        # the self.dependencies(args)
        return Artifact(name, self, args, dependencies_map)


class Artifact(Generic[T], Dependable):
    def __init__(
        self,
        name: str,
        rule: Rule,
        rule_args: T,
        dependencies_map: dict[Dependency, Dependable],
    ):
        self.rule = rule
        self.dependencies_map = dependencies_map
        self.expanded_dependencies_map = {k: k.expand(v) for k, v in self.dependencies_map.items()}
        self.outcomes = self.rule.outcomes(rule_args)
        self.output_map = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: set()
                    )
                )
            )
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
                        nu = come.needs_update(put, ency, able)
                        self.has_outputs_needing_update = self.has_outputs_needing_update or nu
                        if nu or not rule.filter_for_needing_update:
                            self.output_map[put][come][ency].add(
                                (able, nu)
                            )
        # TODO: if an outcome doesn't have outputs, call needs_update with
        # output=None for each dependency/dependable (if there are no
        # dependencies/dependables, use None for those).
        self.needs_update = lambda outcome, output: self.rule.needs_update(
                self.output_map,
                self.rule_args,
                outcome,
                output,
                self.has_outcomes,
                self.has_dependencies,
                has_outputs_needing_update,
        )
        self.executable = self.rule.executable(self.output_map)



GlobExpr = NewType("Glob", str)


class ObjsFromCsArgs(enum.Enum):
    fileName = 0
    fileNameList = 1
    globExpr = 2


GlobDependency = Dependency(
    name="glob",
    dependable_type=GlobExpr,
    expand=lambda x: [FileDependable(f) for f in glob(x)],
)
FileNameDependency = Dependency(
    name="file",
    dependable_type=str,
    expand=lambda x: [FileDependable(x)]
)
FileNameListDependency = Dependency(
    name="files",
    dependable_type=list[str],
    expand=lambda x: [FileDependable(f) for f in x],
)



def objs_from_cs_dependencies(args: ObjsFromCsArgs) -> list[Dependency]:
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
            raise ValueError(f"Path doesn't have an extension: [{spath}]")
        p, e = spath
        if os.sep in e:
            raise ValueError(
                "Path doesn't have an extension (only dot is at the left "
                f"of a {os.sep}): [{spath}]"
            )
        return p + ".o"

    return { FileOutput(extension_to_o(able.filename)): { ency: set((able,)) }
             for ency in deps_to_ables
             for able in deps_to_ables[ency] }

def objs_from_cs_outcomes(args: ObjsFromCsArgs) -> list[Outcome]:
    return [
        Outcome(
            name="obj_files",
            output_type=FileOutput,
            outputs_from_dependables=obj_files_outputs_from_dependables,
            needs_update=lambda out, ency, able: True
        )
    ]

def shell_executable(command: list[str]) -> Executable:
    def execute():
        print("Executing:", command)
        subprocess.check_output(command)
    return execute


def objs_from_cs_executable(output_map: OutputMap):
    ables = [
        ables_and_needs[0]
        for put in output_map
        for come in output_map[put]
        for ency in output_map[put][come]
        for ables_and_needs in output_map[put][come][ency]
    ]

    return shell_executable(
        ["gcc", "-c"] + [able.filename for able in ables]
    )


objs_from_cs = Rule(
    name = "objs_from_cs",
    dependencies = objs_from_cs_dependencies,
    outcomes = objs_from_cs_outcomes,
    executable = objs_from_cs_executable,
)


artifact = objs_from_cs.artifact("this_dir_objs", ObjsFromCsArgs.globExpr, {GlobDependency: "*.c"})
artifact.executable()
print("Outputs", artifact.output_map.keys())
