// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Executables/ParallelInfo/ParallelInfo.hpp"

#include <charm++.h>
#include <ckcallback.h>
#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Math.hpp"

/// \cond HIDDEN_SYMBOLS
ParallelInfo::ParallelInfo(CkArgMsg* msg) {
  // clang-tidy: do not use pointer arithmetic
  Parallel::printf("Executing '%s' using %d processors.\n",
                   msg->argv[0],  // NOLINT
                   Parallel::number_of_procs());
  if (msg->argc > 1) {
    std::stringstream error_msg;
    error_msg << "Expected zero command line options, not " << msg->argc - 1
              << "\nReceived:\n";
    for (int i = 1; i < msg->argc; ++i) {
      // clang-tidy: do not use pointer arithmetic
      error_msg << "'" << msg->argv[i] << "'\n\n";  // NOLINT
    }
    ERROR(error_msg.str());
  }

  Parallel::printf("%s\n", info_from_build());

  Parallel::printf(
      "\nHeader:\n"
      "[1]Node ID            (my_node)\n"
      "[2]PE ID              (my_proc)\n"
      "[3]Number Of Procs    (number_of_procs)\n"
      "[4]Number Of Nodes    (number_of_nodes)\n"
      "[5]Procs On Node      (procs_on_node)\n"
      "[6]My Local Rank      (my_local_rank)\n"
      "[6]First Proc On Node (first_proc_on_node)\n"
      "[7]Node Of This PE    (node_of)\n"
      "[8]Local Rank Of PE   (local_rank_of)\n");
  start_pe_group_check();
}

void ParallelInfo::start_pe_group_check() const {
  Parallel::printf("\n\nStarting Group Check.\n");
  CProxy_PeGroupReporter::ckNew(CkCallback(
      CkIndex_ParallelInfo::start_node_group_check(), this->thisProxy));
}

void ParallelInfo::start_node_group_check() const {
  CProxy_NodeGroupReporter::ckNew(
      CkCallback(CkIndex_ParallelInfo::end_report(), this->thisProxy));
}

[[noreturn]] void ParallelInfo::end_report() const {
  Parallel::printf("\nReport finished.\n");
  Parallel::exit();
}

// Helper print function so that both node and pe group prints can be changed
// easily.
void print_info() {
  const int digits_in_node = number_of_digits(Parallel::number_of_nodes() - 1);
  const int digits_in_pe = number_of_digits(Parallel::number_of_procs() - 1);
  // The format string is generated based on the number of procs and nodes
  // available so that the output is aligned over all nodes and procs
  Parallel::printf(
      "%0"s + std::to_string(digits_in_node) + "d %0" +  // my_node
          std::to_string(digits_in_pe) +
          "d %d %d %0" +  // my_proc, number procs, number nodes
          std::to_string(digits_in_node) +
          "d %03d %d %0" +  // procs_on_node, local rank, first proc
          std::to_string(digits_in_node) + "d %03d\n"s  // this node, local rank
      ,
      Parallel::my_node(), Parallel::my_proc(), Parallel::number_of_procs(),
      Parallel::number_of_nodes(), Parallel::procs_on_node(Parallel::my_node()),
      Parallel::my_local_rank(),
      Parallel::first_proc_on_node(Parallel::my_node()),
      Parallel::node_of(Parallel::my_proc()),
      Parallel::local_rank_of(Parallel::my_proc()));
}

// clang-tidy: google-runtime-references
PeGroupReporter::PeGroupReporter(
    CkCallback& cb_start_node_group_check) {  // NOLINT
  print_info();
  this->contribute(cb_start_node_group_check);
}

// clang-tidy: google-runtime-references
NodeGroupReporter::NodeGroupReporter(CkCallback& cb_end_report) {  // NOLINT
  print_info();
  this->contribute(cb_end_report);
}
/// \endcond

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include "Executables/ParallelInfo/ParallelInfo.def.h"
#pragma GCC diagnostic pop
