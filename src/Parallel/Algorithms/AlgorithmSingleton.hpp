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
 * \brief A Spectre algorithm object that wraps a charm++ singleton chare.
 *
 * \details This object is the definition of the distributed charm++ object
 * associated with the SpECTRE component wrappers (see \ref
 * dev_guide_parallelization_foundations). See also the charm++ documentation:
 * https://charm.readthedocs.io/en/latest/charm++/manual.html
 * When comparing against the implementations in the charm++ documentation, this
 * object is the one directly associated to declarations in the corresponding
 * `AlgorithmSingleton.ci` file.
 *
 * Unlike the techniques described in the charm++ documentation, however, we
 * define `AlgorithmSingleton` as a strong typedef so that the core
 * functionality can be shared between chare types by introducing an additional
 * inheritance layer. The documentation suggests the inheritance path `A ->
 * CBase_A`, where we have the inheritance path:
 * ```
 * AlgorithmSingleton -> AlgorithmImpl -> CBase_AlgorithmSingleton
 * ```
 * That allows us to introduce the template class
 * <code><a href="classParallel_1_1AlgorithmImpl_3_01ParallelComponent_00_01tmpl_1_1list_3_01PhaseDepActionListsPack_8_8_8_01_4_01_4.html">
 * Parallel::AlgorithmImpl</a></code>
 * that handles all generalizable control-flow.
 */
// Note that the above manual link is needed because doxygen can't properly link
// template specializations, but we'd really like to link to the AlgorithmImpl.
template <typename ParallelComponent, typename SpectreArrayIndex>
class AlgorithmSingleton
    : public Parallel::AlgorithmImpl<
          ParallelComponent,
          typename ParallelComponent::phase_dependent_action_list> {
  using algorithm = Parallel::Algorithms::Singleton;

 public:
  explicit AlgorithmSingleton(CkMigrateMessage* message) noexcept
      : Parallel::AlgorithmImpl<
            ParallelComponent,
            typename ParallelComponent::phase_dependent_action_list>{message} {}

  void pup(PUP::er& p) noexcept override {  // NOLINT
    Parallel::AlgorithmImpl<
        ParallelComponent,
        typename ParallelComponent::phase_dependent_action_list>::pup(p);
  }

  using Parallel::AlgorithmImpl<
      ParallelComponent,
      typename ParallelComponent::phase_dependent_action_list>::AlgorithmImpl;
};

#define CK_TEMPLATES_ONLY
#include "Algorithms/AlgorithmSingleton.def.h"
#undef CK_TEMPLATES_ONLY
