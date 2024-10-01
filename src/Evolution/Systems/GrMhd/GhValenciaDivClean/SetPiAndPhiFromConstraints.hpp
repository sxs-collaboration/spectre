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
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace grmhd::GhValenciaDivClean {
/*!
 * \brief Set \f$\Pi_{ab}\f$ from the gauge source function.
 *
 * This is necessary to ensure the initial data is in the desired evolution
 * gauge.
 */
struct SetPiAndPhiFromConstraints {
 public:
  using return_tags = typename gh::gauges::SetPiAndPhiFromConstraints<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3>::return_tags;

  using argument_tags = tmpl::push_back<
      typename gh::gauges::SetPiAndPhiFromConstraints<
          ghmhd::GhValenciaDivClean::InitialData::
              analytic_solutions_and_data_list,
          3>::argument_tags,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::ActiveGrid>;

  using compute_tags = typename gh::gauges::SetPiAndPhiFromConstraints<
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
      3>::compute_tags;
  using const_global_cache_tags =
      typename gh::gauges::SetPiAndPhiFromConstraints<
          ghmhd::GhValenciaDivClean::InitialData::
              analytic_solutions_and_data_list,
          3>::const_global_cache_tags;
  using mutable_global_cache_tags =
      typename gh::gauges::SetPiAndPhiFromConstraints<
          ghmhd::GhValenciaDivClean::InitialData::
              analytic_solutions_and_data_list,
          3>::mutable_global_cache_tags;

  static void apply(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      const double initial_time, const Mesh<3>& dg_mesh,
      const ElementMap<3, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>&
          dg_logical_coordinates,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const gh::gauges::GaugeCondition& gauge_condition,
      const bool set_pi_and_phi_from_constraints, const Mesh<3>& subcell_mesh,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>&
          subcell_logical_coordinates,
      const evolution::dg::subcell::ActiveGrid active_grid) {
    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      gh::gauges::SetPiAndPhiFromConstraints<
          ghmhd::GhValenciaDivClean::InitialData::
              analytic_solutions_and_data_list,
          3>::apply(pi, phi, initial_time, dg_mesh, logical_to_grid_map,
                    grid_to_inertial_map, functions_of_time,
                    dg_logical_coordinates, spacetime_metric, gauge_condition,
                    set_pi_and_phi_from_constraints);
    } else {
      gh::gauges::SetPiAndPhiFromConstraints<
          ghmhd::GhValenciaDivClean::InitialData::
              analytic_solutions_and_data_list,
          3>::apply(pi, phi, initial_time, subcell_mesh, logical_to_grid_map,
                    grid_to_inertial_map, functions_of_time,
                    subcell_logical_coordinates, spacetime_metric,
                    gauge_condition, set_pi_and_phi_from_constraints);
    }
  }
};
}  // namespace grmhd::GhValenciaDivClean
