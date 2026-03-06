// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.FiniteDifference.Derivatives",
                  "[Unit][Evolution]") {
  const size_t points_per_dimension = 5;
  const size_t ghost_zone_size = 3;
  const size_t fd_deriv_order = 4;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) = 1.0;
  }
  const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_hessian{
          subcell_mesh.number_of_grid_points(), 0.0};

  const Element<3> element = TestHelpers::Ccz4::fd::detail::set_element();

  // Testing spacetime_derivatives()
  DirectionalIdMap<3, evolution::dg::subcell::GhostData> all_ghost_data =
      TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          subcell_mesh, logical_coords, element.neighbors(), ghost_zone_size,
          TestHelpers::Ccz4::fd::detail::compute_prim_solution);

  auto volume_prims_for_recons =
      TestHelpers::Ccz4::fd::detail::compute_prim_solution(logical_coords);
  Variables<typename Ccz4::fd::System::variables_tag::tags_list>
      volume_evolved_vars{subcell_mesh.number_of_grid_points()};

  get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(
          volume_prims_for_recons);
  get<::Ccz4::Tags::ATilde<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ATilde<DataVector, 3>>(volume_prims_for_recons);
  get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_prims_for_recons);
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(volume_evolved_vars) =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
          volume_prims_for_recons);
  get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars) =
      get<::Ccz4::Tags::Theta<DataVector>>(volume_prims_for_recons);
  get<::Ccz4::Tags::GammaHat<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::GammaHat<DataVector, 3>>(volume_prims_for_recons);
  get<gr::Tags::Shift<DataVector, 3>>(volume_evolved_vars) =
      get<gr::Tags::Shift<DataVector, 3>>(volume_prims_for_recons);
  get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(
          volume_prims_for_recons);
  get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars) =
      get<gr::Tags::Lapse<DataVector>>(volume_prims_for_recons);

  Variables<
      db::wrap_tags_in<Tags::deriv, typename Ccz4::fd::System::gradients_tags,
                       tmpl::size_t<3>, Frame::Inertial>>
      deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::spacetime_derivatives(
      make_not_null(&deriv_of_Ccz4_vars), volume_evolved_vars, all_ghost_data,
      fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  Variables<
      db::wrap_tags_in<Tags::deriv, typename Ccz4::fd::System::gradients_tags,
                       tmpl::size_t<3>, Frame::Inertial>>
      expected_deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  auto& expected_d_metric =
      get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        expected_d_metric.get(i, j, k) =
            (i == j) ? static_cast<double>(10 * j + 50 * k + 1) : 0.0;
      }
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                         tmpl::size_t<3>, Frame::Inertial>>(
          deriv_of_Ccz4_vars)),
      expected_d_metric);

  auto& expected_d_atilde =
      get<::Tags::deriv<::Ccz4::Tags::ATilde<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        expected_d_atilde.get(i, j, k) =
            i == j ? static_cast<double>(1000 * j + 5000 * k + 1) : 0.0;
      }
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::ATilde<DataVector, 3>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_Ccz4_vars)),
      expected_d_atilde);

  auto& expected_d_conformal_factor =
      get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_conformal_factor.get(i) = 1;
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                         tmpl::size_t<3>, Frame::Inertial>>(
          deriv_of_Ccz4_vars)),
      expected_d_conformal_factor);

  auto& expected_d_trace_extrinsic_curvature =
      get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_trace_extrinsic_curvature.get(i) = 1;
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                         tmpl::size_t<3>, Frame::Inertial>>(
          deriv_of_Ccz4_vars)),
      expected_d_trace_extrinsic_curvature);

  auto& expected_d_theta =
      get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_theta.get(i) = 1;
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_Ccz4_vars)),
      expected_d_theta);

  auto& expected_d_gamma_hat =
      get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_d_gamma_hat.get(i, j) = 1;
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, 3>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_Ccz4_vars)),
      expected_d_gamma_hat);

  auto& expected_d_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_lapse.get(i) = 1;
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_Ccz4_vars)),
      expected_d_lapse);

  auto& expected_d_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_d_shift.get(i, j) = 1;
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                         Frame::Inertial>>(deriv_of_Ccz4_vars)),
      expected_d_shift);

  auto& expected_d_field_b =
      get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          expected_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_d_field_b.get(i, j) = 1;
    }
  }

  CHECK_ITERABLE_APPROX(
      (get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
                         tmpl::size_t<3>, Frame::Inertial>>(
          deriv_of_Ccz4_vars)),
      expected_d_field_b);

  // Testing second_spacetime_derivatives()
  all_ghost_data = TestHelpers::Ccz4::fd::detail::compute_ghost_data(
      subcell_mesh, logical_coords, element.neighbors(), ghost_zone_size,
      TestHelpers::Ccz4::fd::detail::compute_prim_solution_for_second_deriv);

  volume_prims_for_recons =
      TestHelpers::Ccz4::fd::detail::compute_prim_solution_for_second_deriv(
          logical_coords);

  get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(
          volume_prims_for_recons);
  get<::Ccz4::Tags::ATilde<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ATilde<DataVector, 3>>(volume_prims_for_recons);
  get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars) =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_prims_for_recons);
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(volume_evolved_vars) =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
          volume_prims_for_recons);
  get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars) =
      get<::Ccz4::Tags::Theta<DataVector>>(volume_prims_for_recons);
  get<::Ccz4::Tags::GammaHat<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::GammaHat<DataVector, 3>>(volume_prims_for_recons);
  get<gr::Tags::Shift<DataVector, 3>>(volume_evolved_vars) =
      get<gr::Tags::Shift<DataVector, 3>>(volume_prims_for_recons);
  get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(volume_evolved_vars) =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(
          volume_prims_for_recons);
  get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars) =
      get<gr::Tags::Lapse<DataVector>>(volume_prims_for_recons);

  Variables<db::wrap_tags_in<::Tags::second_deriv,
                             typename Ccz4::fd::System::gradients_tags,
                             tmpl::size_t<3>, Frame::Inertial>>
      second_deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::second_spacetime_derivatives(
      make_not_null(&second_deriv_of_Ccz4_vars), volume_evolved_vars,
      all_ghost_data, fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian,
      cell_centered_logical_to_inertial_inv_hessian);

  Variables<db::wrap_tags_in<::Tags::second_deriv,
                             typename Ccz4::fd::System::gradients_tags,
                             tmpl::size_t<3>, Frame::Inertial>>
      expected_second_deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  auto& expected_second_d_metric =
      get<::Tags::second_deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          expected_second_d_metric.get(i, j, k, l) =
              (i == j and i == k)
                  ? static_cast<double>(2 * (10 * k + 50 * l + 1))
                  : 0.0;
        }
      }
    }
  }

  const Approx custom_approx =
      Approx::custom().epsilon(1.0e-12).scale(*std::max_element(
          volume_evolved_vars.data(),
          volume_evolved_vars.data() + volume_evolved_vars.size() - 1));

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::ConformalMetric<DataVector, 3>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_metric, custom_approx);

  auto& expected_second_d_atilde =
      get<::Tags::second_deriv<::Ccz4::Tags::ATilde<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          expected_second_d_atilde.get(i, j, k, l) =
              (i == j and i == k)
                  ? static_cast<double>(2 * (1000 * k + 5000 * l + 1))
                  : 0.0;
        }
      }
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::ATilde<DataVector, 3>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_atilde, custom_approx);

  auto& expected_second_d_conformal_factor =
      get<::Tags::second_deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_second_d_conformal_factor.get(i, j) = i == j ? 2 : 0.0;
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_conformal_factor, custom_approx);

  auto& expected_second_d_trace_extrinsic_curvature =
      get<::Tags::second_deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_second_d_trace_extrinsic_curvature.get(i, j) = i == j ? 2 : 0.0;
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_trace_extrinsic_curvature, custom_approx);

  auto& expected_second_d_theta =
      get<::Tags::second_deriv<::Ccz4::Tags::Theta<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_second_d_theta.get(i, j) = i == j ? 2 : 0.0;
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::Theta<DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_theta, custom_approx);

  auto& expected_second_d_gamma_hat =
      get<::Tags::second_deriv<::Ccz4::Tags::GammaHat<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        expected_second_d_gamma_hat.get(i, j, k) = i == j ? 2 : 0.0;
      }
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::GammaHat<DataVector, 3>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_gamma_hat, custom_approx);

  auto& expected_second_d_lapse =
      get<::Tags::second_deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_second_d_lapse.get(i, j) = i == j ? 2 : 0.0;
      if ((i == 0 and j == 2) or (i == 2 and j == 0)) {
        expected_second_d_lapse.get(i, j) += 1.0;
      }
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                Frame::Inertial>>(second_deriv_of_Ccz4_vars)),
      expected_second_d_lapse, custom_approx);

  auto& expected_second_d_shift =
      get<::Tags::second_deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        expected_second_d_shift.get(i, j, k) = i == j ? 2 : 0.0;
      }
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                                Frame::Inertial>>(second_deriv_of_Ccz4_vars)),
      expected_second_d_shift, custom_approx);

  auto& expected_second_d_field_b =
      get<::Tags::second_deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>>(
          expected_second_deriv_of_Ccz4_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        expected_second_d_field_b.get(i, j, k) = i == j ? 2 : 0.0;
      }
    }
  }

  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Tags::second_deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
                                tmpl::size_t<3>, Frame::Inertial>>(
          second_deriv_of_Ccz4_vars)),
      expected_second_d_field_b, custom_approx);

// Test ASSERT triggers for incorrect neighbor size.
#ifdef SPECTRE_DEBUG
  for (const auto& direction : Direction<3>::all_directions()) {
    const DirectionalId<3> directional_element_id{
        direction, *element.neighbors().at(direction).begin()};
    DirectionalIdMap<3, evolution::dg::subcell::GhostData> bad_ghost_data =
        all_ghost_data;
    DataVector& neighbor_data = bad_ghost_data.at(directional_element_id)
                                    .neighbor_ghost_data_for_reconstruction();
    neighbor_data = DataVector{2};
    const std::string match_string{
        MakeString{}
        << "Amount of reconstruction data sent (" << neighbor_data.size()
        << ") from " << directional_element_id
        << " is not a multiple of the number of reconstruction variables "
        << Variables<Ccz4::fd::Tags::spacetime_reconstruction_tags>::
               number_of_independent_components};
    CHECK_THROWS_WITH(
        Ccz4::fd::spacetime_derivatives(
            make_not_null(&deriv_of_Ccz4_vars), volume_evolved_vars,
            bad_ghost_data, fd_deriv_order, subcell_mesh,
            cell_centered_logical_to_inertial_inv_jacobian),
        Catch::Matchers::ContainsSubstring(match_string));
    CHECK_THROWS_WITH(
        Ccz4::fd::second_spacetime_derivatives(
            make_not_null(&second_deriv_of_Ccz4_vars), volume_evolved_vars,
            bad_ghost_data, fd_deriv_order, subcell_mesh,
            cell_centered_logical_to_inertial_inv_jacobian,
            cell_centered_logical_to_inertial_inv_hessian),
        Catch::Matchers::ContainsSubstring(match_string));
  }
#endif  // SPECTRE_DEBUG
}
