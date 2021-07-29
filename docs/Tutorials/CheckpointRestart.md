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
3. When using `CheckpointAndExitAfterWallclock` to trigger checkpoints, note
   that the elapsed wallclock time is checked only when the `PhaseControl` is
   run, i.e., at global synchronization points defined in the input file.
   This means that to write a checkpoint in the 30 minutes before the end of a
   job's queue time, the triggers in the input file must trigger global
   synchronizations at least once every 30 minutes (and probably 2-3 times so
   there is a margin for the time to write files to disc, etc). It is currently
   up to the user to find the balance between too-frequent synchronizations
   (that slow the code) and too-infrequent synchronizations (that won't allow
   checkpoints to be written).

Certain simulation parameters can be modified when restarting from a checkpoint
file. This is done by parsing a new input file containing just those options to
modify; all other options will preserve their value from the original run.

Note, however, that not all tags are permitted to be modified: in the current
implementation, only tags from the `const_global_cache_tags` that also have a
member variable `static constexpr bool is_overlayable = true;` can be modified.
The reason for this "opt-in" design is that in general, most tags interact with
past or current simulation data in a way that would invalidate the simulation
state if the tag were modified on restart (example: changing the domain
invalidates all spatial data, changing the timestepper invalidates the history).
Only tags that do not interact with the state should be permitted to be updated.
For example: activation thresholds on various algorithms, or frequency of data
observation, are safe parameters to modify.

The executable will update the global cache with new input file values during
the phase `UpdateOptionsAtRestartFromCheckpoint`. The
`CheckpointAndExitAfterWallclock` phase control automatically directs code flow
to this phase after a restart.

TODO FOR CODE REVIEW: If someone uses a
`VisitAndReturn(UpdateOptionsAtRestartFromCheckpoint)` just after a
`VisitAndReturn(WriteCheckpoint)`, the code will try to parse the overlay files
during evolution, even if no restarts occur. In other words, the code might
write a checkpoint every 100 slabs, and will try to reparse the input file after
each one. Since it's hard to know if there was a restart in your past (except
perhaps by making guesses from the wallclock time?), it's hard to handle this
sanely...

In this option-updating phase, the code tries to read an "overlay" input file
whose name is computed from the original input file and the number of the
checkpoint used to restart. Say the original input file is `path/to/Input.yaml`
and the code is restarted using a checkpoint `+restart SpectreCheckpoint000123`,
then the overlay input file to read has name `path/to/Input.overlay000123.yaml`.
If this file does not exist, the executable continues with previous parameter
values.
