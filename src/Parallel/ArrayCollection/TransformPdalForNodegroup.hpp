// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/ArrayCollection/StartPhaseOnNodegroup.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace detail {
template <typename OnePhaseActions>
struct TransformPdalForNodegroup {
  using type = tmpl::conditional_t<
      OnePhaseActions::phase == Parallel::Phase::Initialization, tmpl::list<>,
      Parallel::PhaseActions<OnePhaseActions::phase,
                             tmpl::list<Actions::StartPhaseOnNodegroup>>>;
};
}  // namespace detail

/// \brief Transforms the `PhaseDepActionList` (phase dependent action
/// list/PDAL) from one used for a `evolution::DgElementArray` to that for
/// `Parallel::DgElementCollection`
template <typename PhaseDepActionList>
using TransformPhaseDependentActionListForNodegroup =
    tmpl::flatten<tmpl::transform<PhaseDepActionList,
                                  detail::TransformPdalForNodegroup<tmpl::_1>>>;
}  // namespace Parallel
