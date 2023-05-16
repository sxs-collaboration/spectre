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
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::gauges {
/*!
 * \brief Set \f$\Pi_{ab}\f$ from the gauge source function.
 *
 * This is necessary to ensure the initial data is in the desired evolution
 * gauge.
 */
template <size_t Dim>
struct SetPiFromGauge {
 public:
  using return_tags = tmpl::list<gh::Tags::Pi<DataVector, Dim>>;
  using argument_tags =
      tmpl::list<::Tags::Time, domain::Tags::Mesh<Dim>,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 domain::Tags::FunctionsOfTime,
                 domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
                 gr::Tags::SpacetimeMetric<DataVector, Dim>,
                 gh::Tags::Phi<DataVector, Dim>,
                 gh::gauges::Tags::GaugeCondition>;

  using const_global_cache_tags = tmpl::list<gh::gauges::Tags::GaugeCondition>;

  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
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
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const gauges::GaugeCondition& gauge_condition);
};
}  // namespace gh::gauges
