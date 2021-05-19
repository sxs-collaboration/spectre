// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Utilities/TMPL.hpp"

namespace dg::Events {
template <size_t VolumeDim, typename TimeTag, typename Fields,
          typename SolutionFields>
using field_observations = tmpl::flatten<tmpl::list<
    ObserveFields<VolumeDim, TimeTag, Fields, SolutionFields>,
    tmpl::conditional_t<std::is_same_v<SolutionFields, tmpl::list<>>,
                        tmpl::list<>,
                        ObserveErrorNorms<TimeTag, SolutionFields>>>>;
}  // namespace dg::Events

namespace Events {
template <typename System>
using time_events =
    tmpl::list<Events::ObserveTimeStep<System>, Events::ChangeSlabSize>;
}  // namespace Events
