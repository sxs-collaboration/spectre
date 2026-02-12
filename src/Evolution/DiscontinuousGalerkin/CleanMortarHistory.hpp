// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim, typename T>
class DirectionalIdMap;
class LtsTimeStepper;
namespace Tags {
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace TimeSteppers {
template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
class BoundaryHistory;
}  // namespace TimeSteppers
namespace evolution::dg {
template <size_t Dim>
struct MortarData;
template <size_t Dim>
class MortarInfo;
}  // namespace evolution::dg
namespace evolution::dg::Tags {
template <size_t Dim>
struct MortarDataHistory;
template <size_t Dim>
struct MortarInfo;
}  // namespace evolution::dg::Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg {
/// Mutator to remove old entries from the mortar histories in a
/// local-time-stepping DG evolution.
template <size_t Dim>
struct CleanMortarHistory {
  using return_tags = tmpl::list<evolution::dg::Tags::MortarDataHistory<Dim>>;
  using argument_tags =
      tmpl::list<::Tags::TimeStepper<LtsTimeStepper>, Tags::MortarInfo<Dim>>;

  static void apply(
      gsl::not_null<DirectionalIdMap<
          Dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<Dim>,
                                             ::evolution::dg::MortarData<Dim>,
                                             DataVector>>*>
          history,
      const LtsTimeStepper& time_stepper,
      const DirectionalIdMap<Dim, MortarInfo<Dim>>& mortar_info);
};
}  // namespace evolution::dg
