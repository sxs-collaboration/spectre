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
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiFromGauge.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean {
/*!
 * \brief Set \f$\Pi_{ab}\f$ from the gauge source function.
 *
 * This is necessary to ensure the initial data is in the desired evolution
 * gauge.
 */
struct SetPiFromGauge {
 public:
  using return_tags = tmpl::list<gh::Tags::Pi<DataVector, 3>>;
  using argument_tags = tmpl::list<
      ::Tags::Time, domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      domain::Tags::ElementMap<3, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTime,
      domain::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      gr::Tags::SpacetimeMetric<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
      gh::gauges::Tags::GaugeCondition,
      evolution::dg::subcell::Tags::ActiveGrid>;

  using const_global_cache_tags = tmpl::list<gh::gauges::Tags::GaugeCondition>;

  static void apply(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      const double initial_time, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh,
      const ElementMap<3, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>&
          dg_logical_coordinates,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>&
          subcell_logical_coordinates,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
      const gh::gauges::GaugeCondition& gauge_condition,
      const evolution::dg::subcell::ActiveGrid active_grid) {
    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      gh::gauges::SetPiFromGauge<3>::apply(
          pi, initial_time, dg_mesh, logical_to_grid_map, grid_to_inertial_map,
          functions_of_time, dg_logical_coordinates, spacetime_metric, phi,
          gauge_condition);
    } else {
      gh::gauges::SetPiFromGauge<3>::apply(
          pi, initial_time, subcell_mesh, logical_to_grid_map,
          grid_to_inertial_map, functions_of_time, subcell_logical_coordinates,
          spacetime_metric, phi, gauge_condition);
    }
  }
};
}  // namespace grmhd::GhValenciaDivClean
