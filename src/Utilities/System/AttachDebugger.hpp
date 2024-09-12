// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace sys {
/*!
 * \brief Provide an infinite loop to attach a debugger during startup. Useful
 * for debugging MPI runs.
 *
 * Each MPI rank writes a file name `spectre_pid_#_host_NAME` to the working
 * directory. This allows you to attach GDB to the running process using
 * `gdb --pid=PID`, once for each MPI rank. You must then halt the program
 * using `C-c` and then call `set var i = 7` inside GDB. Once you've done this
 * on each MPI rank, you can have each MPI rank `continue`.
 *
 * To add support for attaching to a debugger in an executable, you must add
 * `sys::attach_debugger` to the
 * `Parallel::charmxx::register_init_node_and_proc` init node functions. Then,
 * when you launch the executable launch it as
 * ```shell
 * SPECTRE_ATTACH_DEBUGGER=1 mpirun -np N ...
 * ```
 * The environment variable `SPECTRE_ATTACH_DEBUGGER` being set tells the code
 * to allow attaching from a debugger.
 */
void attach_debugger();
}  // namespace sys
