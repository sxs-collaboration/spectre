// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// This file should be included in each file which defines a singleton
/// parallel component; doing so ensures that the correct charm++ chares are
/// defined for executables that use that parallel component.

#pragma once

#include "Parallel/Algorithm.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/ArrayIndex.hpp"

/*!
 * \ingroup ParallelGroup
 * \brief A Spectre algorithm object that wraps a single element charm++ array
 * chare.
 *
 * \details Charm++ does offer a distributed object called a singleton, however
 * we don't use this for a few reasons:
 *
 * 1. Charm++ singletons cannot be (easily) placed on particular processors.
 *    Typically we will want singletons on their own processors.
 * 2. Charm++ singletons don't participate in load balancing.
 * 3. Charm++ singletons don't participate in checkpoint/restart when restarted
 *    on a different number of procs.
 *
 * For implementation details, see AlgorithmArray.
 */
template <typename ParallelComponent, typename SpectreArrayIndex>
class AlgorithmSingleton
    : public Parallel::AlgorithmImpl<
          ParallelComponent,
          typename ParallelComponent::phase_dependent_action_list> {
  using algorithm = Parallel::Algorithms::Singleton;

 public:
  using Parallel::AlgorithmImpl<
      ParallelComponent,
      typename ParallelComponent::phase_dependent_action_list>::AlgorithmImpl;
};

#include "Parallel/Algorithms/AlgorithmArray.hpp"
