// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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
template <size_t Dim, typename CouplingResult>
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
template <typename System>
struct CleanMortarHistory {
  static constexpr size_t dim = System::volume_dim;
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
  using CouplingResult = typename dt_variables_tag::type;

  using return_tags =
      tmpl::list<evolution::dg::Tags::MortarDataHistory<dim, CouplingResult>>;
  using argument_tags =
      tmpl::list<::Tags::TimeStepper<LtsTimeStepper>, Tags::MortarInfo<dim>>;

  static void apply(
      gsl::not_null<DirectionalIdMap<
          dim,
          TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<dim>,
                                        ::evolution::dg::MortarData<dim>,
                                        math_wrapper_type<CouplingResult>>>*>
          history,
      const LtsTimeStepper& time_stepper,
      const DirectionalIdMap<dim, MortarInfo<dim>>& mortar_info);
};
}  // namespace evolution::dg
