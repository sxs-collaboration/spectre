// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares the chares for ParallelInfo

#pragma once

#include "Executables/ParallelInfo/ParallelInfo.decl.h"
#include "Parallel/Reduction.hpp"

/// \cond
class CkArgMsg;
class CkCallback;
/// \endcond

/*!
 * \page ParallelInfoExecutablePage ParallelInfo Executable
 * \tableofcontents
 * The %ParallelInfo executable can be used to check the number of nodes, and
 * processing elements (roughly number of cores) registered with Charm++.
 * Depending on the build of Charm++ the way Charm++ identifies nodes will vary.
 * Specifically, for non-SMP builds of Charm++ each processing element is
 * identified as a node instead of each physical node being identified as a
 * node. The substantially increased number of "nodes" for non-SMP builds can
 * cause problems if large amounts of read-only data is cached on a per-node
 * basis.
 *
 * The ParallelInfo executable starts one process on each processing element and
 * each node, which then all print identifying information such as the
 * processing element ID, node ID, etc.
 */

/// \cond HIDDEN_SYMBOLS
class ParallelInfo : public CBase_ParallelInfo {
 public:
  explicit ParallelInfo(CkArgMsg* msg);
  void start_node_group_check() const;
  [[noreturn]] void end_report() const;

 private:
  void start_pe_group_check() const;
};

class PeGroupReporter : public Group {
 public:
  // clang-tidy: non-const reference, Charm++ interface
  explicit PeGroupReporter(CkCallback& cb_start_node_group_check);  // NOLINT
};

class NodeGroupReporter : public NodeGroup {
 public:
  // clang-tidy: non-const reference, Charm++ interface
  explicit NodeGroupReporter(CkCallback& cb_end_report);  // NOLINT
};
/// \endcond
