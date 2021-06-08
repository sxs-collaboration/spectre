\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# %Setting up checkpoints and restarts {#tutorial_checkpoint_restart}

SpECTRE executables can write checkpoints that save their instantaneous state to
disc; the execution can be restarted later from a saved checkpoint. This feature
is useful for expensive simulations that would run longer than the wallclock
limits on a supercomputer system.

Executables can checkpoint when:
1. The `Phase` enumeration in the `Metavariables` has a `WriteCheckpoint` phase.
2. The `WriteCheckpoint` phase is run by a `PhaseControl` specified in the
   `Metavariables` and the input file. The two supported ways of running the
   checkpoint phase are:
   - with `CheckpointAndExitAfterWallclock`. This is the recommended phase
     control for checkpointing, because it writes only one checkpoint before
     cleanly terminating the code.
     This reduces the disc space taken up by checkpoint files and stops using
     up the allocation's CPU-hours on work that would be redone anyway after the
     run is restarted.
   - using `VisitAndReturn(WriteCheckpoint)`. This is useful for writing more
     frequent checkpoint files, which could help when debugging a run by
     restarting it from just before the failure.

To restart an executable from a checkpoint file, run a command like this:
```
./MySpectreExecutable +restart SpectreCheckpoint000123
```
where the `000123` should be the number of the checkpoint to restart from.

There are a number of caveats in the current implementation of checkpointing
and restarting:

1. The same binary must be used when writing the checkpoint and when restarting
   from the checkpoint. If a different binary is used to restart the code,
   there are no guarantees that the code will restart or that the continued
   execution will be correct.
2. The code must be restarted on the same hardware configuration used when
   writing the checkpoint --- this means the same number of nodes with the same
   number of processors per node.
3. Currently, there is no support for modifying any parameters during a restart.
   The restart only extends a simulation's runtime beyond wallclock limits.
4. When using `CheckpointAndExitAfterWallclock` to trigger checkpoints, note
   that the elapsed wallclock time is checked only when the `PhaseControl` is
   run, i.e., at global synchronization points defined in the input file.
   This means that to write a checkpoint in the 30 minutes before the end of a
   job's queue time, the triggers in the input file must trigger global
   synchronizations at least once every 30 minutes (and probably 2-3 times so
   there is a margin for the time to write files to disc, etc). It is currently
   up to the user to find the balance between too-frequent synchronizations
   (that slow the code) and too-infrequent synchronizations (that won't allow
   checkpoints to be written).

