import abc
import enum
from collections import defaultdict
from glob import glob
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Generic, Optional, Type, TypeVar, Union


T = TypeVar("T")


class Dependable(abc.ABC):
    pass


class TimestampedDependable(Dependable):
    @abc.abstractmethod
    def timestamp(self) -> Optional[float]:
        pass


class AtomicDependable(Dependable):
    pass


class FileDependable(TimestampedDependable, AtomicDependable):
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


class Output(AtomicDependable):
    @abc.abstractmethod
    def needs_update(
        self,
        dependency: Optional[Dependency] = None,
        dependable: Optional[AtomicDependable] = None,
    ):
        pass


class FileOutput(Output, FileDependable):
    def __init__(self, filename: str):
        super().__init__(filename)

    def needs_update(
        self,
        dependency: Optional[Dependency] = None,
        dependable: Optional[TimestampedDependable] = None,
    ):
        return timestamp_based_outcome(self, dependency, dependable)


def _default_outcome_needs_update(
    output: Optional[Output],
    dependency: Optional[Dependency] = None,
    dependable: Optional[AtomicDependable] = None,
) -> bool:
    # Outcomes without outputs always trigger.
    if output is None:
        return True
    return output.needs_update(dependency, dependable)


AtomicAble = TypeVar("AtomicAble", bound=AtomicDependable)

@dataclass(frozen=True)
class Outcome(Generic[T, AtomicAble]):
    name: str
    output_type: Optional[Type[T]]
    outputs_from_dependables: Callable[
        [dict[Dependency, list[AtomicAble]]],
        dict[Output, dict[Dependency, list[AtomicAble, bool]]],
    ]
    needs_update: Callable[
        [Optional[T], Optional[Dependency], Optional[AtomicAble]],
        bool
    ] = _default_outcome_needs_update

    @staticmethod
    def from_args(o: Callable[[T], "Outcome"]):
        return OutcomeFromArgs(o)


@dataclass(frozen=True)
class OutcomeFromArgs(Generic[T]):
    outcomeFromArgs: Callable[[T], Outcome]


Put = TypeVar("Put", bound=Output)
Come = TypeVar("Come", bound=Outcome)
Ency = TypeVar("Ency", bound=Dependency)
Able = TypeVar("Able", bound=Dependable)


class OutputMap(Generic[Put, Come, Ency, Able]):
    def __init__(self, output_map: dict[Put, dict[Able, dict[bool, dict[Come, set[Ency]]]]]):
        self.output_map = output_map

    def __call__(self) -> dict[Put, dict[Able, dict[bool, dict[Come, set[Ency]]]]]:
        return self.output_map


Action = Callable[[], None]


@dataclass
class Executable:
    executable: Callable[[OutputMap], Action]

    @staticmethod
    def from_args(instruction: Callable[[T], "Executable"]) -> "Instruction":
        return Instruction(instruction)


@dataclass
class Instruction(Generic[T]):
    instruction: Callable[[T], Executable]


DependenciesExpander = Callable[
    [dict[Dependency, Dependable]],
    dict[Dependency, list[AtomicDependable]],
]


class PlanNode:
    def __init__(self, artifact: "Artifact"):
        self.artifact = artifact
        self.dependencies: list[PlanNode] = []
        self.needed_by: list[PlanNode] = []
        self.done: bool = False

    def add_dependency(self, plan_node: "PlanNode"):
        self.dependencies.append(plan_node)
        plan_node.needed_by.append(self)


def _default_rule_needs_update(output_map: OutputMap, _come: Outcome, _put: Output) -> bool:
    return any(
        nu
        for put in output_map()
        for able in output_map()[put]
        for nu in output_map()[put][able]
    )


class Rule(Generic[T]):
    def __init_subclass__(
            cls,
            # name,
            # bases,
            # attrs,
            dependencies: Optional[Callable[[T], list[Dependency]]] = None,
            executable: Optional[Callable[[T], Executable]] = None,
            needs_update: Callable[[OutputMap], bool] = _default_rule_needs_update,
            filter_for_needing_update: bool = True,
    ):
        # _executable = None
        _outcomes = []
        print("Dict:", cls.__dict__)
        for k, v in cls.__dict__.items():
            if isinstance(v, Outcome):
                # TODO: make _outcomes a dictionary?
                _outcomes.append(OutcomeFromArgs(lambda _, vv=v: vv))
            elif isinstance(v, OutcomeFromArgs):
                # TODO
                _outcomes.append(v)

        def new_init(_c, *_args, **_kwargs):
            raise ValueError("Pake rules can't be instantiated")
        cls.__init__ = new_init
        cls._executable = executable
        cls._dependencies = dependencies
        cls._outcomes = _outcomes
        cls._needs_update = needs_update
        cls._filter_for_needing_update = filter_for_needing_update
        # return t
        # raise ValueError("Pake rules can't be instantiated")
        # print(attrs)
        # return type.__new__(mcs, name, bases, attrs)
        # print("abcABCabc")
        # for a in cls.__dict__:
        #     print(a)

    @classmethod
    def outcomes(cls) -> list[OutcomeFromArgs]:
        return getattr(cls, "_outcomes")

    @classmethod
    def dependencies(cls):
        return getattr(cls, "_dependencies")

    @classmethod
    def needs_update(cls):
        return getattr(cls, "_needs_update")

    @classmethod
    def executable(cls):
        return getattr(cls, "_executable")


def artifact(
    rule: Type[Rule],
    name: str,
    args: T,
    dependencies_map: dict[Dependency, Dependable]
) -> "Artifact":
    # TODO: Check that the keys of the map coincide with
    # the self.dependencies(args)
    return Artifact(name, rule, args, dependencies_map)


class ArtifactOutputs(AtomicDependable, Generic[Put]):
    def __init__(self, artifact: "Artifact", outputs: set[Put]):
        self.artifact = artifact
        self.outputs = outputs


def expand_maybe_artifact_outputs(dependency, dependable):
    if isinstance(dependable, ArtifactOutputs):
        # TODO: can artifact outputs be expanded?
        return [ArtifactOutputs(dependable.artifact, {put}) for put in dependable.outputs]
    return dependency.expand(dependable)


class Artifact(Dependable, Generic[T]):
    def __init__(
        self,
        name: str,
        rule: type[Rule],
        rule_args: T,
        dependencies_map: dict[Dependency, Dependable],
    ):
        self.name = name
        self.rule = rule
        self.rule_args = rule_args
        self.dependencies_map = dependencies_map
        self.outcomes = [res for o in self.rule.outcomes() if (res := o.outcomeFromArgs(rule_args)) is not None]

    def get_output_map(self):
        output_map = OutputMap(defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: set()
                    )
                )
            )
        ))

        expanded_dependencies_map = {
            k: expand_maybe_artifact_outputs(k, v)
            for k, v in self.dependencies_map.items()
        }

        for come in self.outcomes:
            for put, ency_to_ables in come.outputs_from_dependables(
                expanded_dependencies_map
            ).items():
                for ency, ables in ency_to_ables.items():
                    for able in ables:
                        if isinstance(able, ArtifactOutputs):
                            for able_output in able.outputs:
                                nu = come.needs_update(put, ency, able_output)
                                print("Dependable from artifact:", able_output, "nu:", nu)
                                output_map()[put][able_output][nu][come].add(ency)
                        else:
                            nu = come.needs_update(put, ency, able)
                            print("Dependable:", able, "nu:", nu)
                            output_map()[put][able][nu][come].add(ency)

        # TODO: if an outcome doesn't have outputs, call needs_update with
        # output=None for each dependency/dependable (if there are no
        # dependencies/dependables, use None for those).

        return output_map

    def get_executable(self) -> Optional[Callable]:
        output_map = self.get_output_map()
        if not self.needs_update(output_map):
            return None
        return self.rule.executable().executable(output_map)

    def outputs(self, outcome_name: str) -> AtomicDependable:
        puts = set()
        output_map = self.get_output_map()
        for put in output_map():
            for able in output_map()[put]:
                # TODO: add some parameter to specify whether
                #  to get them all or only the updated ones?
                # TODO: this is happening at the wrong time, we shouldn't
                #  get the OutputMap until we're ready to execute.
                #  Perhaps make that a property of the PlanNode?
                for nu in output_map()[put][able]:
                    for come in output_map()[put][able][nu]:
                        if come.name == outcome_name:
                            puts.add(put)
        return ArtifactOutputs(self, puts)

    def needs_update(self, output_map: OutputMap, outcome: Optional[Outcome] = None, output: Optional[Output] = None):
        return self.rule.needs_update()(
            output_map,
            outcome,
            output,
        )

    def expand_plan_node(
        self,
        plan: "Plan",
    ) -> PlanNode:
        artifact = self

        output_map = artifact.get_output_map()
        execute = False
        plan_node = PlanNode(artifact)
        
        for able in self.dependencies_map.values():
            print("able in expand: ", able)
            if isinstance(able, ArtifactOutputs):
                dep_plan_node = plan.get_or_generate_plan_node(able.artifact)
                plan_node.add_dependency(dep_plan_node)
                # TODO: assuming that if a dependency executed, you must execute.
                execute = True

        return plan_node

    def make(self):
        plan = Plan()
        plan.get_or_generate_plan_node(self)
        plan.execute()


class Plan:
    def __init__(self):
        self.leaf_nodes: list[PlanNode] = []
        self.upstream: list[Artifact] = []
        # Use None for artifacts that are not executed.
        self.artifact_to_node: dict[Artifact, Optional[PlanNode]] = {}

    def execute(self):
        finished: dict[int, PlanNode] = {}
        max_finished = 0

        def execute_node(n):
            nonlocal max_finished
            executable = n.artifact.get_executable()
            if executable is not None:
                executable()
                n.done = True
                if len(n.needed_by) > 0:
                    finished[max_finished] = n
                    max_finished += 1
 
        for leaf_node in self.leaf_nodes:
            execute_node(leaf_node)

        while len(finished) != 0:
            all_done = []
            for i, f in finished.items():
                for upstream in f.needed_by:
                    if upstream.done:
                        continue
                    if all(d.done for d in upstream.dependencies):
                        execute_node(upstream)
                if all(u.done for u in f.needed_by):
                    all_done.append(i)
            for i in all_done:
                del finished[i]

    def check_circular_dependency(self, artifact: Artifact):
        for n in self.upstream:
            if n.name == artifact.name:
                raise ValueError(
                    "Circular dependency:\n"
                    + "\n->".join(n.name for n in self.upstream)
                    + "\n->"
                    + artifact.name
                )

    def get_or_generate_plan_node(self, artifact: Artifact) -> Optional[PlanNode]:
        if artifact in self.artifact_to_node:
            return self.artifact_to_node[artifact]
        self.check_circular_dependency(artifact)
        self.upstream.append(artifact)
        plan_node = artifact.expand_plan_node(self)
        assert self.upstream[-1] is artifact
        self.upstream.pop()
        self.artifact_to_node[artifact] = plan_node
        if plan_node is not None and len(plan_node.dependencies) == 0:
            self.leaf_nodes.append(plan_node)
        return plan_node


@dataclass
class GlobExpr(Dependable):
    globs: list[str]


class ObjsFromCsArgs(enum.Enum):
    fileName = 0
    fileNameList = 1
    globExpr = 2


GlobDependency = Dependency(
    name="glob",
    dependable_type=GlobExpr,
    expand=lambda globs: [FileDependable(f) for g in globs.globs for f in glob(g)],
)


FileNameDependency = Dependency(
    name="file",
    dependable_type=str,
    expand=lambda x: [FileDependable(x)]
)


def expand_file_name_list(x: Union[list[str], ArtifactOutputs]):
    if isinstance(x, ArtifactOutputs):
        return ArtifactOutputs(x.artifact, {FileDependable(f.filename) for f in x.outputs})
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
        return [FileNameDependency]
    if args is ObjsFromCsArgs.fileNameList:
        return [FileNameListDependency]


def timestamp_based_outcome(
    out: TimestampedDependable,
    ency: Dependency,
    able: TimestampedDependable
) -> bool:
    print("out:", out)
    print("ency:", ency)
    print("able:", able)
    if out.timestamp() is None:
        return True
    if able.timestamp() is None:
        return True
    return able.timestamp() >= out.timestamp()


def _do_execute_in_shell(command: list[str]):
    print("Executing:", command)
    subprocess.check_output(command)


def shell_executable(command: list[str]) -> Action:
    return lambda: _do_execute_in_shell(command)


def dependables_needing_update(output_map: OutputMap[Put, Come, Ency, Able]) -> list[Able]:
    om = output_map()
    return [
        able
        for put in om
        for able in om[put]
        if True in om[put][able]
    ]


def single_file_output_from_dependables(filename: str, deps_to_ables: dict[Dependency, list[AtomicDependable]]):
    return {
        FileOutput(filename): {
            ency: set([able for able in deps_to_ables[ency]])
            for ency in deps_to_ables
        }
    }


def file_globs_dependency(globs: list[str]) -> dict[Dependency, Dependable]:
    return {GlobDependency: GlobExpr(globs)}
