import abc
import enum
from glob import glob
from dataclasses import dataclass
import os
import subprocess
from typing import Callable, Generic, List, NewType, Optional, Type, TypeVar, Union


T = TypeVar("T")


class Dependable:
    def needs_update() -> bool:
        return True


class AtomicDependable(Dependable):
    pass


@dataclass
class FileDependable(AtomicDependable):
    filename: str


@dataclass
class Output:
    pass
    

@dataclass
class FileOutput(Output):
    filename: str

@dataclass(frozen=True)
class Dependency(Generic[T]):
    name: str
    dependable_type: Type[T]
    expand: Optional[Callable[[T], list[AtomicDependable]]]
    is_optional: bool = False


@dataclass
class Outcome(Generic[T]):
    name: str
    output_type: T
    outputs_from_dependables: Callable[
        [dict[Dependency, list[AtomicDependable]]],
        list[T],
    ]


Executable = Callable[[], None]


DependenciesExpander = Callable[
    dict[Dependency, Dependable],
    dict[Dependency, list[AtomicDependable]],
]


@dataclass
class Rule(Generic[T]):
    name: str
    dependencies: Callable[[T], list[Dependency]]
    outcomes: Callable[[T], list[Outcome]]
    executable: Callable[[T], Executable]

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
        self.executable = self.rule.executable(self.expanded_dependencies_map)
        self.outputs = [puts for comes in self.rule.outcomes(rule_args) for puts in comes.outputs_from_dependables(self.expanded_dependencies_map)]

GlobExpr = NewType("Glob", "str")


class ObjsFromCsArgs(enum.Enum):
    fileName = 0
    fileNameList = 1
    globExpr = 2


GlobDependency = Dependency(
    name="glob",
    dependable_type=GlobExpr,
    expand=lambda x: [FileDependable(f) for f in glob(x)],
)
FileDependency = Dependency(
    name="file",
    dependable_type=str,
    expand=lambda x: [FileDependable(x)],
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


def objs_from_cs_outcomes(args: ObjsFromCsArgs) -> list[Outcome]:
    def extension_to_o(path):
        spath = path.rsplit(".", 1)
        if len(spath) < 2:
            raise ValueError(f"Path doesn't have an extension: [{path}]")
        p, e = spath
        if os.sep in e:
            raise ValueError(
                "Path doesn't have an extension (only dot is at the left "
                f"of a {os.sep}): [{path}]"
            )
        return p + ".o"

    if args in (ObjsFromCsArgs.globExpr, ObjsFromCsArgs.fileNameList):
        return [
            Outcome(
                name="obj_files",
                output_type=FileOutput,
                outputs_from_dependables=lambda d: [
                    FileOutput(filename=extension_to_o(f.filename))
                    for _, dables in d.items()
                    for f in dables
                ]
            )
        ]
    return [
        Outcome(
            name="obj_file",
            output_type=FileOutput,
            outputs_from_dependables=lambda d: [
                FileOutput(filename=extension_to_o(list(d.values())[0].filename))
            ]
        )
    ]

def shell_executable(command: list[str]) -> Executable:
    def execute():
        print("Executing:", command)
        subprocess.check_output(command)
    return execute


def objs_from_cs_executable(deps: dict[Dependency, list[AtomicDependable]]):
    print("Deps:", deps)
    return shell_executable(
        ["gcc", "-c"] + [f.filename for ds in deps.values() for f in ds]
    )


objs_from_cs = Rule(
    name = "objs_from_cs",
    dependencies = objs_from_cs_dependencies,
    outcomes = objs_from_cs_outcomes,
    executable = objs_from_cs_executable,
)


artifact = objs_from_cs.artifact("this_dir_objs", ObjsFromCsArgs.globExpr, {GlobDependency: "*.c"})
artifact.executable()
print("Outputs", artifact.outputs)
