// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Informer.

#pragma once

/// \cond
class CkArgMsg;
/// \endcond

/// \ingroup LoggingGroup
/// The Informer manages textual output regarding the status of a simulation.
class Informer {
 public:
  /// Print useful information at the beginning of a simulation.
  ///
  /// This includes the command used to start the executable such as
  ///
  /// ```
  /// ./MyExecutable --input-file MyInputFile.yaml
  /// ```
  ///
  /// If you used charmrun, mpirun, or something similar to start your
  /// executable, you'll only see the options that have to do with the
  /// executable itself. Meaning, for this command
  ///
  /// ```
  /// mpirun -np 4 MyExecutable --input-file MyInputFile.yaml
  /// ```
  ///
  /// only `MyExecutable` and onwards will be printed.
  static void print_startup_info(CkArgMsg* msg);

  /// Print useful information at the end of a simulation.
  static void print_exit_info();
};
