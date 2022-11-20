# Pake, make anything using dependencies

Pake allows to specify a set of rules, and artifacts
that can be built using those rules, perhaps using the outcomes
of other artifacts as inputs/dependencies.

In can be used as a build system, a workflow execution tool,
or for whatever code you can specify using dependencies (that is,
pretty much anything).

All code is written in Python, thus avoiding the need to learn
new syntax, escaping, etc. when using a language specific for,
say, a build system.

Plans for extension include:
- Pre-built rules for common usage patterns.
- Multi-threading.
- Support for running in containers.
- A web interface to explore the state of a run.
- Support for scheduling.
- Wiring to web frameworks (for instance, modelling the response
to a web request as an artifact, with the computation of the request
split as the computation of several intermediate artifacts).


## Usage

The following code defines two artifacts:

```
        class ThisDirObjs(Artifact):
            rule = ObjsFromCs
            args = ObjsFromCsArgs.globExpr
            deps = {
                GlobDependency: GlobExpr(["*.c"])
            }

        class ThisDirExec(Artifact):
            rule = ExecFromObjs
            args = ExecFromObjsArgs(exec_name="ppp")
            deps = {
                FileNameListDependency: ThisDirObjs.outputs("obj_files")
            }

        ThisDirExec.make()
```

The artifact named ThisDirObjs uses the ObjsFromCs rule to build object
files for each C file in the directory. These object files are consumed
by the artifact named ThisDirExec and produce an executable file called
`ppp`.

The line with `make()` triggers the instructions to produce the files.
It works as you would expect as a build system, rebuilding object files
only if the corresponding C files have changed, and rebuilding the
executable only of some of the object files have changed.
