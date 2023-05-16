// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// [[TimeOut, 10]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Derivatives",
    "[Unit][Evolution]") {
  const size_t points_per_dimension = 5;
  const size_t ghost_zone_size = 3;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords = TestHelpers::grmhd::GhValenciaDivClean::fd::
      detail::set_logical_coordinates(subcell_mesh);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) = 1.0;
  }

  const Element<3> element =
      TestHelpers::grmhd::GhValenciaDivClean::fd::detail::set_element();

  const FixedHashMap<maximum_number_of_neighbors(3),
                     std::pair<Direction<3>, ElementId<3>>,
                     evolution::dg::subcell::GhostData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      all_ghost_data = TestHelpers::grmhd::GhValenciaDivClean::fd::detail::
          compute_ghost_data(subcell_mesh, logical_coords, element.neighbors(),
                             ghost_zone_size,
                             TestHelpers::grmhd::GhValenciaDivClean::fd::
                                 detail::compute_prim_solution);
  const auto volume_prims_for_recons =
      TestHelpers::grmhd::GhValenciaDivClean::fd::detail::compute_prim_solution(
          logical_coords);
  Variables<
      typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>
      volume_evolved_vars{subcell_mesh.number_of_grid_points()};
  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_evolved_vars) =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(volume_prims_for_recons);
  get<gh::Tags::Phi<DataVector, 3>>(volume_evolved_vars) =
      get<gh::Tags::Phi<DataVector, 3>>(volume_prims_for_recons);
  get<gh::Tags::Pi<DataVector, 3>>(volume_evolved_vars) =
      get<gh::Tags::Pi<DataVector, 3>>(volume_prims_for_recons);

  Variables<db::wrap_tags_in<
      Tags::deriv, typename grmhd::GhValenciaDivClean::System::gradients_tags,
      tmpl::size_t<3>, Frame::Inertial>>
      deriv_of_gh_vars{subcell_mesh.number_of_grid_points()};

  grmhd::GhValenciaDivClean::fd::spacetime_derivatives(
      make_not_null(&deriv_of_gh_vars), volume_evolved_vars, all_ghost_data,
      subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

  Variables<db::wrap_tags_in<
      Tags::deriv, typename grmhd::GhValenciaDivClean::System::gradients_tags,
      tmpl::size_t<3>, Frame::Inertial>>
      expected_deriv_of_gh_vars{subcell_mesh.number_of_grid_points()};

  auto& expected_d_metric =
      get<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          expected_deriv_of_gh_vars);
  get<0, 0, 0>(expected_d_metric) = get<1, 0, 0>(expected_d_metric) =
      get<2, 0, 0>(expected_d_metric) = 0.0;
  get<0, 1, 0>(expected_d_metric) = get<1, 1, 0>(expected_d_metric) =
      get<2, 1, 0>(expected_d_metric) = 0.001;
  get<0, 2, 0>(expected_d_metric) = get<1, 2, 0>(expected_d_metric) =
      get<2, 2, 0>(expected_d_metric) = 0.002;
  get<0, 3, 0>(expected_d_metric) = get<1, 3, 0>(expected_d_metric) =
      get<2, 3, 0>(expected_d_metric) = 0.003;
  get<0, 1, 1>(expected_d_metric) = get<1, 1, 1>(expected_d_metric) =
      get<2, 1, 1>(expected_d_metric) = 0.002;
  get<0, 2, 1>(expected_d_metric) = get<1, 2, 1>(expected_d_metric) =
      get<2, 2, 1>(expected_d_metric) = 0.004;
  get<0, 2, 2>(expected_d_metric) = get<1, 2, 2>(expected_d_metric) =
      get<2, 2, 2>(expected_d_metric) = 0.006;
  get<0, 3, 1>(expected_d_metric) = get<1, 3, 1>(expected_d_metric) =
      get<2, 3, 1>(expected_d_metric) = 0.006;
  get<0, 2, 2>(expected_d_metric) = get<1, 2, 2>(expected_d_metric) =
      get<2, 2, 2>(expected_d_metric) = 0.006;
  get<0, 3, 2>(expected_d_metric) = get<1, 3, 2>(expected_d_metric) =
      get<2, 3, 2>(expected_d_metric) = 0.009;
  get<0, 3, 3>(expected_d_metric) = get<1, 3, 3>(expected_d_metric) =
      get<2, 3, 3>(expected_d_metric) = 0.012;

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                         tmpl::size_t<3>, Frame::Inertial>>(deriv_of_gh_vars)),
      expected_d_metric);

  auto& expected_d_pi =
      get<::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_gh_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {
        expected_d_pi.get(i, a, b) = 500.0 * a + 10000.0 * b + 1.0 + i;
      }
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_gh_vars)),
      expected_d_pi);

  auto& expected_d_phi =
      get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_gh_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t a = 0; a < 4; ++a) {
        for (size_t b = a; b < 4; ++b) {
          expected_d_phi.get(i, j, a, b) =
              i == j ? (50.0 * a + 1000.0 * b + 10.0 * i + 1.0) : 0.0;
        }
      }
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_gh_vars)),
      expected_d_phi);

  // Test ASSERT triggers for incorrect neighbor size.
#ifdef SPECTRE_DEBUG
  for (const auto& direction : Direction<3>::all_directions()) {
    const std::pair directional_element_id{
        direction, *element.neighbors().at(direction).begin()};
    FixedHashMap<maximum_number_of_neighbors(3),
                 std::pair<Direction<3>, ElementId<3>>,
                 evolution::dg::subcell::GhostData,
                 boost::hash<std::pair<Direction<3>, ElementId<3>>>>
        bad_ghost_data = all_ghost_data;
    DataVector& neighbor_data = bad_ghost_data.at(directional_element_id)
                                    .neighbor_ghost_data_for_reconstruction();
    neighbor_data = DataVector{2};
    const std::string match_string{
        MakeString{}
        << "Amount of reconstruction data sent (" << neighbor_data.size()
        << ") from " << directional_element_id
        << " is not a multiple of the number of reconstruction variables "
        << Variables<grmhd::GhValenciaDivClean::Tags::
                         primitive_grmhd_and_spacetime_reconstruction_tags>::
               number_of_independent_components};
    CHECK_THROWS_WITH(grmhd::GhValenciaDivClean::fd::spacetime_derivatives(
                          make_not_null(&deriv_of_gh_vars), volume_evolved_vars,
                          bad_ghost_data, subcell_mesh,
                          cell_centered_logical_to_inertial_inv_jacobian),
                      Catch::Contains(match_string));
  }
#endif  // SPECTRE_DEBUG
}
