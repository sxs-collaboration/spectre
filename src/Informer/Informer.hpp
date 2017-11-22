// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Informer.

#pragma once

#include <charm++.h>

/// \ingroup UtilitiesGroup
/// The Informer manages textual output regarding the status of a simulation.
class Informer {
 public:
  /// Print useful information at the beginning of a simulation.
  static void print_startup_info(CkArgMsg* msg);

  /// Print useful information at the end of a simulation.
  static void print_exit_info();
};
