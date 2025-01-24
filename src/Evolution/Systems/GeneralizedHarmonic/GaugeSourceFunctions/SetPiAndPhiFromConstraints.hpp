// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace gh {
namespace Tags {
/// DataBox tag for holding whether or not to set GH variables $\Pi$ and $\Phi$
/// from constraints.
struct SetPiAndPhiFromConstraints : db::SimpleTag {
  using type = bool;

  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;

  static bool create_from_options() { return true; }
};
}  // namespace Tags

namespace gauges {
/*!
 * \brief GlobalCache mutator to set the value of the
 * `gh::Tags::SetPiAndPhiFromConstraints` tag.
 */
struct SetPiAndPhiFromConstraintsCacheMutator {
  static void apply(gsl::not_null<bool*> value, bool new_value);
};

/*!
 * \brief Set \f$\Pi_{ab}\f$ from the gauge source function (or 1-index
 * constraint) and \f$\Phi_{iab}\f$ from the 3-index constraint.
 *
 * This is necessary to ensure the initial data is in the desired evolution
 * gauge and that the 1- and 3-index constraints are satisfied.
 */
template <class AllSolutionsForChristoffelAnalytic, size_t Dim>
struct SetPiAndPhiFromConstraints {
 public:
  using return_tags =
      tmpl::list<gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>>;
  using argument_tags =
      tmpl::list<::Tags::Time, domain::Tags::Mesh<Dim>,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                 gr::Tags::SpacetimeMetric<DataVector, Dim>,
                 gh::gauges::Tags::GaugeCondition,
                 gh::Tags::SetPiAndPhiFromConstraints>;

  using compute_tags = tmpl::list<
      Parallel::Tags::FromGlobalCache<gh::Tags::SetPiAndPhiFromConstraints>>;
  using const_global_cache_tags = tmpl::list<gh::gauges::Tags::GaugeCondition>;
  using mutable_global_cache_tags =
      tmpl::list<gh::Tags::SetPiAndPhiFromConstraints>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*> phi,
      double time, const Mesh<Dim>& mesh,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
          logical_coordinates,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const gauges::GaugeCondition& gauge_condition,
      bool set_pi_and_phi_from_constraints = true);
};
}  // namespace gauges
}  // namespace gh
