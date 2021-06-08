// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using frame = Frame::Inertial;
constexpr size_t VolumeDim = 3;

// Test boundary conditions on dt<VSpacetimeMetric> in 3D against SpEC
void test_constraint_preserving_bjorhus_v_psi_vs_spec_3d(
    const size_t grid_size_each_dimension) noexcept {
  // Setup grid
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();

  // Populate various tensors needed to compute BcDtVSpacetimeMetric
  tnsr::iaa<DataVector, VolumeDim, frame> local_three_index_constraint(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::I<DataVector, VolumeDim, frame> local_unit_interface_normal_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::array<DataVector, 4> local_char_speeds{
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN())};
  // Allocate memory for output
  tnsr::aa<DataVector, VolumeDim, frame> local_bc_dt_v_psi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  {
    // Setting the 3-index constraint
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          // In SpEC, this constraint is explicitly computed using
          // d_i psi_ab and phi_iab as inputs. The explicit subtractions
          // below are here to remind us what values those two input
          // tensors were set to in SpEC to get the desired BC in this test.
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 0.;
      get<2>(local_unit_interface_normal_vector)[i] = 0.;
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
    }

    // Compute rhs value
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_bjorhus_corrections_dt_v_psi(
            make_not_null(&local_bc_dt_v_psi),
            local_unit_interface_normal_vector, local_three_index_constraint,
            local_char_speeds);
    // Setting local_RhsVSpacetimeMetric
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(0, a)[i] += 23.;
      }
      for (size_t a = 1; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(1, a)[i] += 29.;
      }
      for (size_t a = 2; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(2, a)[i] += 31.;
      }
      local_bc_dt_v_psi.get(3, 3)[i] += 37.;
    }
  }
  // Initialize with values from SpEC
  auto spec_bc_dt_v_psi =
      make_with_value<tnsr::aa<DataVector, VolumeDim, frame>>(
          local_bc_dt_v_psi, std::numeric_limits<double>::signaling_NaN());

  for (size_t i = 0; i < slice_grid_points; ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      spec_bc_dt_v_psi.get(0, a)[i] = 25.4;
    }
    get<1, 1>(spec_bc_dt_v_psi)[i] = 31.4;
    get<1, 2>(spec_bc_dt_v_psi)[i] = 31.4;
    get<1, 3>(spec_bc_dt_v_psi)[i] = 31.4;
    get<2, 2>(spec_bc_dt_v_psi)[i] = 33.4;
    get<2, 3>(spec_bc_dt_v_psi)[i] = 33.4;
    get<3, 3>(spec_bc_dt_v_psi)[i] = 39.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_psi, spec_bc_dt_v_psi);

  // Test for another set of values
  {
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 1.;
      get<2>(local_unit_interface_normal_vector)[i] = 1.;
    }
    // Compute rhs value
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_bjorhus_corrections_dt_v_psi(
            make_not_null(&local_bc_dt_v_psi),
            local_unit_interface_normal_vector, local_three_index_constraint,
            local_char_speeds);
    // Setting local_RhsVSpacetimeMetric
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(0, a)[i] += 23.;
      }
      for (size_t a = 1; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(1, a)[i] += 29.;
      }
      for (size_t a = 2; a <= VolumeDim; ++a) {
        local_bc_dt_v_psi.get(2, a)[i] += 31.;
      }
      local_bc_dt_v_psi.get(3, 3)[i] += 37.;
    }
  }

  // Initialize with values from SpEC
  for (size_t i = 0; i < slice_grid_points; ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      spec_bc_dt_v_psi.get(0, a)[i] = 20.;
    }
    get<1, 1>(spec_bc_dt_v_psi)[i] = 26.;
    get<1, 2>(spec_bc_dt_v_psi)[i] = 26.;
    get<1, 3>(spec_bc_dt_v_psi)[i] = 26.;
    get<2, 2>(spec_bc_dt_v_psi)[i] = 28.;
    get<2, 3>(spec_bc_dt_v_psi)[i] = 28.;
    get<3, 3>(spec_bc_dt_v_psi)[i] = 34.;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_psi, spec_bc_dt_v_psi);
}

// Test boundary conditions on dt<VZero>
void test_constraint_preserving_bjorhus_v_zero_vs_spec_3d(
    const size_t grid_size_each_dimension) noexcept {
  // Setup grid
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();

  // Populate various tensors needed to compute BcDtVZero
  tnsr::iaa<DataVector, VolumeDim, frame> local_four_index_constraint(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::I<DataVector, VolumeDim, frame> local_unit_interface_normal_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  std::array<DataVector, 4> local_char_speeds{
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN())};
  // Allocate memory for output
  tnsr::iaa<DataVector, VolumeDim, frame> local_bc_dt_v_zero(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  {
    // Setting the 4-index constraint:
    // initialize dPhi (with same values as for SpEC) and compute C4 from it
    auto local_dphi = make_with_value<tnsr::ijaa<DataVector, VolumeDim, frame>>(
        local_unit_interface_normal_vector,
        std::numeric_limits<double>::signaling_NaN());
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        for (size_t i = 0; i < slice_grid_points; ++i) {
          local_dphi.get(0, 0, a, b)[i] = 3.;
          local_dphi.get(0, 1, a, b)[i] = 5.;
          local_dphi.get(0, 2, a, b)[i] = 7.;
          local_dphi.get(1, 0, a, b)[i] = 59.;
          local_dphi.get(1, 1, a, b)[i] = 61.;
          local_dphi.get(1, 2, a, b)[i] = 67.;
          local_dphi.get(2, 0, a, b)[i] = 73.;
          local_dphi.get(2, 1, a, b)[i] = 79.;
          local_dphi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // C4_{iab} = LeviCivita^{ijk} dphi_{jkab}
    local_four_index_constraint =
        GeneralizedHarmonic::four_index_constraint(local_dphi);

    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 1.;
      get<2>(local_unit_interface_normal_vector)[i] = 1.;
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
    }

    // Compute rhs value
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_bjorhus_corrections_dt_v_zero(
            make_not_null(&local_bc_dt_v_zero),
            local_unit_interface_normal_vector, local_four_index_constraint,
            local_char_speeds);
    // Setting local_Rhs
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          local_bc_dt_v_zero.get(0, a, b)[i] += 91.;
          local_bc_dt_v_zero.get(1, a, b)[i] += 97.;
          local_bc_dt_v_zero.get(2, a, b)[i] += 101.;
        }
      }
    }
  }
  // Initialize with values from SpEC
  auto spec_bc_dt_v_zero =
      make_with_value<tnsr::iaa<DataVector, VolumeDim, frame>>(
          local_bc_dt_v_zero, std::numeric_limits<double>::signaling_NaN());

  for (size_t i = 0; i < slice_grid_points; ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      spec_bc_dt_v_zero.get(0, 0, a)[i] = 79.;
      spec_bc_dt_v_zero.get(1, 0, a)[i] = 90.4;
      spec_bc_dt_v_zero.get(2, 0, a)[i] = 95.6;
    }
    for (size_t a = 1; a <= VolumeDim; ++a) {
      spec_bc_dt_v_zero.get(0, 1, a)[i] = 79.;
      spec_bc_dt_v_zero.get(1, 1, a)[i] = 90.4;
      spec_bc_dt_v_zero.get(2, 1, a)[i] = 95.6;
    }

    get<0, 2, 2>(spec_bc_dt_v_zero)[i] = 79.;
    get<0, 2, 3>(spec_bc_dt_v_zero)[i] = 79.;
    get<1, 2, 2>(spec_bc_dt_v_zero)[i] = 90.4;
    get<1, 2, 3>(spec_bc_dt_v_zero)[i] = 90.4;
    get<2, 2, 2>(spec_bc_dt_v_zero)[i] = 95.6;
    get<2, 2, 3>(spec_bc_dt_v_zero)[i] = 95.6;

    get<0, 3, 3>(spec_bc_dt_v_zero)[i] = 79.;
    get<1, 3, 3>(spec_bc_dt_v_zero)[i] = 90.4;
    get<2, 3, 3>(spec_bc_dt_v_zero)[i] = 95.6;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_zero, spec_bc_dt_v_zero);

  // Test for another set of values
  {
    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 0.;
      get<2>(local_unit_interface_normal_vector)[i] = 0.;
    }
    // Compute rhs value
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_bjorhus_corrections_dt_v_zero(
            make_not_null(&local_bc_dt_v_zero),
            local_unit_interface_normal_vector, local_four_index_constraint,
            local_char_speeds);
    // Setting local_Rhs
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          local_bc_dt_v_zero.get(0, a, b)[i] += 91.;
          local_bc_dt_v_zero.get(1, a, b)[i] += 97.;
          local_bc_dt_v_zero.get(2, a, b)[i] += 101.;
        }
      }
    }
  }

  // Initialize with values from SpEC
  for (size_t i = 0; i < slice_grid_points; ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      spec_bc_dt_v_zero.get(0, 0, a)[i] = 91.;
      spec_bc_dt_v_zero.get(1, 0, a)[i] = 91.6;
      spec_bc_dt_v_zero.get(2, 0, a)[i] = 94.4;
    }
    for (size_t a = 1; a <= VolumeDim; ++a) {
      spec_bc_dt_v_zero.get(0, 1, a)[i] = 91.;
      spec_bc_dt_v_zero.get(1, 1, a)[i] = 91.6;
      spec_bc_dt_v_zero.get(2, 1, a)[i] = 94.4;
    }

    get<0, 2, 2>(spec_bc_dt_v_zero)[i] = 91.;
    get<0, 2, 3>(spec_bc_dt_v_zero)[i] = 91.;
    get<1, 2, 2>(spec_bc_dt_v_zero)[i] = 91.6;
    get<1, 2, 3>(spec_bc_dt_v_zero)[i] = 91.6;
    get<2, 2, 2>(spec_bc_dt_v_zero)[i] = 94.4;
    get<2, 2, 3>(spec_bc_dt_v_zero)[i] = 94.4;

    get<0, 3, 3>(spec_bc_dt_v_zero)[i] = 91.;
    get<1, 3, 3>(spec_bc_dt_v_zero)[i] = 91.6;
    get<2, 3, 3>(spec_bc_dt_v_zero)[i] = 94.4;
  }

  // Compare values returned by BC action vs those from SpEC
  CHECK_ITERABLE_APPROX(local_bc_dt_v_zero, spec_bc_dt_v_zero);
}

void test_constraint_preserving_physical_bjorhus_v_minus_vs_spec_3d(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& /* upper_bound */) noexcept {
  // Setup grid
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  // Setup coordinates
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, VolumeDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        get<0>(tmp)[i * VolumeDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * VolumeDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();

  // Populate various tensors needed to compute BcDtVMinus as done in SpEC
  tnsr::I<DataVector, VolumeDim, frame> local_unit_interface_normal_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::I<DataVector, VolumeDim, frame> local_inertial_coords(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  const auto& local_inertial_coords_x = get<0>(local_inertial_coords);
  const auto& local_inertial_coords_y = get<1>(local_inertial_coords);
  const auto& local_inertial_coords_z = get<2>(local_inertial_coords);
  Scalar<DataVector> local_constraint_gamma2(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // timelike and spacelike SPACETIME vectors, l^a and k^a
  tnsr::a<DataVector, VolumeDim, frame> local_outgoing_null_one_form(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::a<DataVector, VolumeDim, frame> local_incoming_null_one_form(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // timelike and spacelike SPACETIME oneforms, l_a and k_a
  tnsr::A<DataVector, VolumeDim, frame> local_outgoing_null_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::A<DataVector, VolumeDim, frame> local_incoming_null_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // spacetime projection operator P_ab, P^ab, and P^a_b
  tnsr::AA<DataVector, VolumeDim, frame> local_projection_AB(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::aa<DataVector, VolumeDim, frame> local_projection_ab(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::Ab<DataVector, VolumeDim, frame> local_projection_Ab(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // constraint characteristics
  tnsr::a<DataVector, VolumeDim, frame> local_constraint_char_zero_minus(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::a<DataVector, VolumeDim, frame> local_constraint_char_zero_plus(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // RhsVSpacetimeMetric and RhsVMinus
  tnsr::aa<DataVector, VolumeDim, frame> local_char_projected_rhs_dt_v_psi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::aa<DataVector, VolumeDim, frame> local_char_projected_rhs_dt_v_minus(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // Vars
  tnsr::aa<DataVector, VolumeDim, frame> local_pi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::iaa<DataVector, VolumeDim, frame> local_phi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // Char speeds
  std::array<DataVector, 4> local_char_speeds{
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN()),
      DataVector(slice_grid_points,
                 std::numeric_limits<double>::signaling_NaN())};

  // interface normal one form
  tnsr::i<DataVector, VolumeDim, frame> local_unit_interface_normal_one_form(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // spacetime unit normal vec
  tnsr::A<DataVector, VolumeDim, frame> local_spacetime_unit_normal_vector(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // metrics
  tnsr::II<DataVector, VolumeDim, frame> local_inverse_spatial_metric(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::aa<DataVector, VolumeDim, frame> local_spacetime_metric(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::AA<DataVector, VolumeDim, frame> local_inverse_spacetime_metric(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // extrinsic curvature
  tnsr::ii<DataVector, VolumeDim, frame> local_extrinsic_curvature(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // deriv of pi and phi
  tnsr::iaa<DataVector, VolumeDim, frame> local_d_pi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());
  tnsr::ijaa<DataVector, VolumeDim, frame> local_d_phi(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  // 3-index constraint
  tnsr::iaa<DataVector, VolumeDim, frame> local_three_index_constraint(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  {
    // Setting coords
    for (size_t i = 0; i < VolumeDim; ++i) {
      local_inertial_coords.get(i) = inertial_coords.get(i);
    }

    // Setting local_unit_interface_normal_one_form
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_unit_interface_normal_one_form.get(0)[i] = -1.;
      local_unit_interface_normal_one_form.get(1)[i] = 1.;
      local_unit_interface_normal_one_form.get(2)[i] = 1.;
    }
    // Setting local_spacetime_unit_normal_vector
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_spacetime_unit_normal_vector.get(0)[i] = -1.;
      local_spacetime_unit_normal_vector.get(1)[i] = -3.;
      local_spacetime_unit_normal_vector.get(2)[i] = -5.;
      local_spacetime_unit_normal_vector.get(3)[i] = -7.;
    }
    // Setting local_inverse_spatial_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        local_inverse_spatial_metric.get(0, j)[i] = 41.;
        local_inverse_spatial_metric.get(1, j)[i] = 43.;
        local_inverse_spatial_metric.get(2, j)[i] = 47.;
      }
    }
    // Setting local_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_spacetime_metric.get(0, a)[i] = 257.;
        local_spacetime_metric.get(1, a)[i] = 263.;
        local_spacetime_metric.get(2, a)[i] = 269.;
        local_spacetime_metric.get(3, a)[i] = 271.;
      }
    }
    // Setting local_inverse_spacetime_metric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_inverse_spacetime_metric.get(0, a)[i] =
            -277.;  // needs to be < 0 for lapse
        local_inverse_spacetime_metric.get(1, a)[i] = 281.;
        local_inverse_spacetime_metric.get(2, a)[i] = 283.;
        local_inverse_spacetime_metric.get(3, a)[i] = 293.;
      }
    }
    // Setting local_extrinsic_curvature
    // ONLY ON THE +Y AXIS (Y = +0.5)
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      std::array<double, 9> spec_vals{};
      bool not_initialized = true;

      if ((local_inertial_coords_x[i] == 299. or
           local_inertial_coords_x[i] == 299.5 or
           local_inertial_coords_x[i] == 300.) and
          local_inertial_coords_y[i] == 0.5 and
          (local_inertial_coords_z[i] == -0.5 or
           local_inertial_coords_z[i] == 0. or
           local_inertial_coords_z[i] == 0.5)) {
        spec_vals = {{200.2198037251189, 266.7930716334918, 333.3663395418648,
                      266.7930716334918, 333.3663395418648, 399.9396074502377,
                      333.3663395418648, 399.9396074502377, 466.5128753586106}};
        not_initialized = false;
      }

      if (not_initialized) {
        ERROR("Not checking the correct face, coordinates not recognized");
      }
      for (size_t j = 0; j < VolumeDim; ++j) {
        for (size_t k = 0; k < VolumeDim; ++k) {
          local_extrinsic_curvature.get(j, k)[i] =
              gsl::at(spec_vals, j * (0 + VolumeDim) + k);
        }
      }
    }

    // Setting local_d_phi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_d_pi.get(0, a, b)[i] = 1.;
          local_d_phi.get(0, 0, a, b)[i] = 3.;
          local_d_phi.get(0, 1, a, b)[i] = 5.;
          local_d_phi.get(0, 2, a, b)[i] = 7.;
          local_d_pi.get(1, a, b)[i] = 53.;
          local_d_phi.get(1, 0, a, b)[i] = 59.;
          local_d_phi.get(1, 1, a, b)[i] = 61.;
          local_d_phi.get(1, 2, a, b)[i] = 67.;
          local_d_pi.get(2, a, b)[i] = 71.;
          local_d_phi.get(2, 0, a, b)[i] = 73.;
          local_d_phi.get(2, 1, a, b)[i] = 79.;
          local_d_phi.get(2, 2, a, b)[i] = 83.;
        }
      }
    }
    // Setting 3idxConstraint
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_three_index_constraint.get(0, a, b)[i] = 11. - 3.;
          local_three_index_constraint.get(1, a, b)[i] = 13. - 5.;
          local_three_index_constraint.get(2, a, b)[i] = 17. - 7.;
        }
      }
    }

    // Setting constraint_gamma2
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      get(local_constraint_gamma2)[i] = 113.;
    }
    // Note: explicit division by sqrt(2) below for the computation of ui, uI,
    // vi, vI are left as-is to remind us that SpEC uses these null vectors
    // and one forms without normalization by sqrt(2).
    // Setting incoming null one_form: ui
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_one_form.get(0)[i] = -2. / sqrt(2.);
      local_incoming_null_one_form.get(1)[i] = 5. / sqrt(2.);
      local_incoming_null_one_form.get(2)[i] = 3. / sqrt(2.);
      local_incoming_null_one_form.get(3)[i] = 7. / sqrt(2.);
    }
    // Setting incoming null vector: uI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_incoming_null_vector.get(0)[i] = -1. / sqrt(2.);
      local_incoming_null_vector.get(1)[i] = 13. / sqrt(2.);
      local_incoming_null_vector.get(2)[i] = 17. / sqrt(2.);
      local_incoming_null_vector.get(3)[i] = 19. / sqrt(2.);
    }
    // Setting outgoing null one_form: vi
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_one_form.get(0)[i] = -1. / sqrt(2.);
      local_outgoing_null_one_form.get(1)[i] = 3. / sqrt(2.);
      local_outgoing_null_one_form.get(2)[i] = 2. / sqrt(2.);
      local_outgoing_null_one_form.get(3)[i] = 5. / sqrt(2.);
    }
    // Setting outgoing null vector: vI
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      local_outgoing_null_vector.get(0)[i] = -1. / sqrt(2.);
      local_outgoing_null_vector.get(1)[i] = 2. / sqrt(2.);
      local_outgoing_null_vector.get(2)[i] = 3. / sqrt(2.);
      local_outgoing_null_vector.get(3)[i] = 5. / sqrt(2.);
    }
    // Setting projection Ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_Ab.get(0, a)[i] = 233.;
        local_projection_Ab.get(1, a)[i] = 239.;
        local_projection_Ab.get(2, a)[i] = 241.;
        local_projection_Ab.get(3, a)[i] = 251.;
      }
    }
    // Setting projection ab
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_ab.get(0, a)[i] = 379.;
        local_projection_ab.get(1, a)[i] = 383.;
        local_projection_ab.get(2, a)[i] = 389.;
        local_projection_ab.get(3, a)[i] = 397.;
      }
    }
    // Setting projection AB
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_projection_AB.get(0, a)[i] = 353.;
        local_projection_AB.get(1, a)[i] = 359.;
        local_projection_AB.get(2, a)[i] = 367.;
        local_projection_AB.get(3, a)[i] = 373.;
      }
    }
    // Setting local_RhsVSpacetimeMetric
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_v_psi.get(0, a)[i] = 23.;
        local_char_projected_rhs_dt_v_psi.get(1, a)[i] = 29.;
        local_char_projected_rhs_dt_v_psi.get(2, a)[i] = 31.;
        local_char_projected_rhs_dt_v_psi.get(3, a)[i] = 37.;
      }
    }
    // Setting RhsVMinus
    for (size_t i = 0; i < get<0>(local_inertial_coords).size(); ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        local_char_projected_rhs_dt_v_minus.get(0, a)[i] = 331.;
        local_char_projected_rhs_dt_v_minus.get(1, a)[i] = 337.;
        local_char_projected_rhs_dt_v_minus.get(2, a)[i] = 347.;
        local_char_projected_rhs_dt_v_minus.get(3, a)[i] = 349.;
      }
    }

    // Setting unit_interface_normal_Vector
    for (size_t i = 0; i < slice_grid_points; ++i) {
      get<0>(local_unit_interface_normal_vector)[i] = -1.;
      get<1>(local_unit_interface_normal_vector)[i] = 1.;
      get<2>(local_unit_interface_normal_vector)[i] = 1.;
    }
    // Setting pi AND phi
    for (size_t i = 0; i < slice_grid_points; ++i) {
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = 0; b <= VolumeDim; ++b) {
          local_pi.get(a, b)[i] = 1.;
          local_phi.get(0, a, b)[i] = 3.;
          local_phi.get(1, a, b)[i] = 5.;
          local_phi.get(2, a, b)[i] = 7.;
        }
      }
    }
    // Setting char speeds
    for (size_t i = 0; i < slice_grid_points; ++i) {
      local_char_speeds.at(0)[i] = -0.3;
      local_char_speeds.at(1)[i] = -0.1;
      local_char_speeds.at(3)[i] = -0.2;
    }
    // Setting constraint_char_zero_plus AND constraint_char_zero_minus
    // ONLY ON THE +Y AXIS (Y = +0.5)
    for (size_t i = 0; i < slice_grid_points; ++i) {
      std::array<double, 4> spec_vals{};
      std::array<double, 4> spec_vals2{};
      bool not_initialized = true;

      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{3722388974.799386, -16680127.68747905, -22991565.68745775,
                      -29394198.68743645}};
        spec_vals2 = {{3866802572.424386, -19802442.06247905,
                       -19496408.06245775, -19257249.06243645}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{1718287695.133032, -35082292.16184025, -44420849.32723039,
                      -53850596.45928719}};
        spec_vals2 = {{1866652790.548975, -38170999.90741873,
                       -40892085.07280888, -43680040.20486567}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{1718299060.415949, -35082089.27527188, -44420613.14790046,
                      -53850326.98725116}};
        spec_vals2 = {{1866664134.129611, -38170797.39904065,
                       -40891849.27166924, -43679771.11101994}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{1718276282.501221, -35082495.89577083, -44421086.49296889,
                      -53850867.05677793}};
        spec_vals2 = {{1866641399.709844, -38171203.26157932,
                       -40892321.85877737, -43680310.4225864}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{1718287685.660846, -35082292.33093269, -44420849.52407002,
                      -53850596.68387395}};
        spec_vals2 = {{1866652781.094877, -38171000.07619599,
                       -40892085.26933331, -43680040.42913724}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{1718299050.990844, -35082089.44352208, -44420613.34375966,
                      -53850327.21071925}};
        spec_vals2 = {{1866664124.722503, -38170797.56697725,
                       -40891849.46721483, -43679771.33417442}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{1718276292.082241, -35082495.72473276, -44421086.29386414,
                      -53850866.82960649}};
        spec_vals2 = {{1866641409.272568, -38171203.09086005,
                       -40892321.65999144, -43680310.19573379}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{1718287695.194063, -35082292.16074976, -44420849.32596073,
                      -53850596.45783833}};
        spec_vals2 = {{1866652790.609889, -38170999.90633029,
                       -40892085.07154126, -43680040.20341886}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{1718299060.476573, -35082089.27418864, -44420613.14663924,
                      -53850326.98581193}};
        spec_vals2 = {{1866664134.190119, -38170797.39795944,
                       -40891849.27041004, -43679771.10958273}};
        not_initialized = false;
      }
      if (not_initialized) {
        ERROR("Not checking the correct face, coordinates not recognized");
      }
      for (size_t j = 0; j <= VolumeDim; ++j) {
        local_constraint_char_zero_plus.get(j)[i] = gsl::at(spec_vals, j);
        local_constraint_char_zero_minus.get(j)[i] = gsl::at(spec_vals2, j);
      }
    }
  }

  // Memory for output
  tnsr::aa<DataVector, VolumeDim, frame> local_bc_dt_v_minus(
      slice_grid_points, std::numeric_limits<double>::signaling_NaN());

  {
    // Compute the new boundary corrections for dt<vminus>
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_bjorhus_corrections_dt_v_minus(
            make_not_null(&local_bc_dt_v_minus), local_constraint_gamma2,
            local_inertial_coords, local_incoming_null_one_form,
            local_outgoing_null_one_form, local_incoming_null_vector,
            local_outgoing_null_vector, local_projection_ab,
            local_projection_Ab, local_projection_AB,
            local_char_projected_rhs_dt_v_psi,
            local_char_projected_rhs_dt_v_minus,
            local_constraint_char_zero_plus, local_constraint_char_zero_minus,
            local_char_speeds);
    // Add in the current boundary value to get corrected values for dt<vminus>
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        local_bc_dt_v_minus.get(a, b) +=
            local_char_projected_rhs_dt_v_minus.get(a, b);
      }
    }

    // Initialize with values from SpEC
    tnsr::aa<DataVector, VolumeDim, frame> spec_bc_dt_v_minus(
        slice_grid_points, std::numeric_limits<double>::signaling_NaN());

    for (size_t i = 0; i < slice_grid_points; ++i) {
      std::array<double, 16> spec_vals{};
      bool not_initialized = true;

      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{-22857869555.93848, 330934979077.5689, 242482973593.2169,
                      507808643438.4713, 330934979077.5689, 690190606335.8783,
                      600780523621.8837, 868984678067.689, 242482973593.2169,
                      600780523621.8837, 514052337348.0055, 781537245156.9633,
                      507808643438.4713, 868984678067.689, 781537245156.9633,
                      1054439575483.625}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{6085210488.394159, 167380193629.8459, 127052654518.3203,
                      248004925243.5957, 167380193629.8459, 332680231531.8318,
                      291581334988.8124, 414851930920.4022, 127052654518.3203,
                      291581334988.8124, 252051387928.7005, 374742777072.0203,
                      248004925243.5957, 414851930920.4022, 374742777072.0203,
                      501009779843.9042}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{6084984067.712499, 167381093626.4053, 127053272910.3827,
                      248006388447.6548, 167381093626.4053, 332682261170.2852,
                      291583083105.2007, 414854523601.7003, 127053272910.3827,
                      291583083105.2007, 252052859833.8237, 374745093603.8436,
                      248006388447.6548, 414854523601.7003, 374745093603.8436,
                      501012947925.7576}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{6085437853.704128, 167379289884.4027, 127052033550.7529,
                      248003455943.9015, 167379289884.4027, 332678193437.6569,
                      291579579589.7132, 414849327437.366, 127052033550.7529,
                      291579579589.7132, 252049909891.7733, 374740450889.0901,
                      248003455943.9015, 414849327437.366, 374740450889.0901,
                      501006598562.8735}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{6085210677.100476, 167380192879.7604, 127052654002.9328,
                      248004924024.1152, 167380192879.7604, 332680229840.2671,
                      291581333531.8772, 414851928759.5795, 127052654002.9328,
                      291581333531.8772, 252051386701.9684, 374742775141.3494,
                      248004924024.1152, 414851928759.5795, 374742775141.3494,
                      501009777203.5247}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{6084984255.479748, 167381092880.0475, 127053272397.5563,
                      248006387234.2355, 167381092880.0475, 332682259487.1282,
                      291583081655.5068, 414854521451.6181, 127053272397.5563,
                      291583081655.5068, 252052858613.1887, 374745091682.769,
                      248006387234.2355, 414854521451.6181, 374745091682.769,
                      501012945298.5021}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{6085437662.827703, 167379290643.1056, 127052034072.0609,
                      248003457177.3931, 167379290643.1056, 332678195148.6575,
                      291579581063.3884, 414849329623.0162, 127052034072.0609,
                      291579581063.3884, 252049911132.6, 374740452841.9442,
                      248003457177.3931, 414849329623.0162, 374740452841.9442,
                      501006601233.5914}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{6085210487.177534, 167380193634.6784, 127052654521.6405,
                      248004925251.4529, 167380193634.6784, 332680231542.7307,
                      291581334998.1996, 414851930934.3248, 127052654521.6405,
                      291581334998.1996, 252051387936.6043, 374742777084.4599,
                      248004925251.4529, 414851930934.3248, 374742777084.4599,
                      501009779860.9169}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{6084984066.503965, 167381093631.2057, 127053272913.6808,
                      248006388455.4596, 167381093631.2057, 332682261181.1116,
                      291583083114.5252, 414854523615.53, 127053272913.6808,
                      291583083114.5252, 252052859841.6748, 374745093616.2003,
                      248006388455.4596, 414854523615.53, 374745093616.2003,
                      501012947942.6568}};
        not_initialized = false;
      }
      if (not_initialized) {
        ERROR("Not checking the correct face, coordinates not recognized");
      }
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_v_minus.get(a, b)[i] =
              gsl::at(spec_vals, a * (1 + VolumeDim) + b);
        }
      }
    }

    // Compare values returned by BC action vs those from SpEC
    CHECK_ITERABLE_APPROX(local_bc_dt_v_minus, spec_bc_dt_v_minus);
  }

  {  // Compute the new boundary corrections for dt<vminus>
    GeneralizedHarmonic::BoundaryConditions::Bjorhus::
        constraint_preserving_physical_bjorhus_corrections_dt_v_minus(
            make_not_null(&local_bc_dt_v_minus), local_constraint_gamma2,
            local_inertial_coords, local_unit_interface_normal_one_form,
            local_unit_interface_normal_vector,
            local_spacetime_unit_normal_vector, local_incoming_null_one_form,
            local_outgoing_null_one_form, local_incoming_null_vector,
            local_outgoing_null_vector, local_projection_ab,
            local_projection_Ab, local_projection_AB,
            local_inverse_spatial_metric, local_extrinsic_curvature,
            local_spacetime_metric, local_inverse_spacetime_metric,
            local_three_index_constraint, local_char_projected_rhs_dt_v_psi,
            local_char_projected_rhs_dt_v_minus,
            local_constraint_char_zero_plus, local_constraint_char_zero_minus,
            local_phi, local_d_phi, local_d_pi, local_char_speeds);
    // Add in the current boundary value to get corrected values for dt<vminus>
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        local_bc_dt_v_minus.get(a, b) +=
            local_char_projected_rhs_dt_v_minus.get(a, b);
      }
    }

    // Initialize with values from SpEC
    tnsr::aa<DataVector, VolumeDim, frame> spec_bc_dt_v_minus(
        slice_grid_points, std::numeric_limits<double>::signaling_NaN());

    for (size_t i = 0; i < slice_grid_points; ++i) {
      std::array<double, 16> spec_vals{};
      bool not_initialized = true;

      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{7.122948293721196e+19, 7.122948329100482e+19,
                      7.122948320255282e+19, 7.122948346787851e+19,
                      7.122948329100482e+19, 7.639071460057165e+19,
                      7.639071451116156e+19, 7.639071477936572e+19,
                      7.122948320255282e+19, 7.639071451116156e+19,
                      8.413256084990011e+19, 8.413256111738503e+19,
                      7.122948346787851e+19, 7.639071477936572e+19,
                      8.413256111738503e+19, 9.445502329090969e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{7.122948296615504e+19, 7.122948312745003e+19,
                      7.122948308712251e+19, 7.122948320807479e+19,
                      7.122948312745003e+19, 7.639071424306128e+19,
                      7.639071420196236e+19, 7.639071432523298e+19,
                      7.122948308712251e+19, 7.639071420196236e+19,
                      8.413256058789916e+19, 8.413256071059056e+19,
                      7.122948320807479e+19, 7.639071432523298e+19,
                      8.413256071059056e+19, 9.445502273747989e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == -0.5) {
        spec_vals = {{7.122948296615481e+19, 7.122948312745094e+19,
                      7.122948308712312e+19, 7.122948320807626e+19,
                      7.122948312745094e+19, 7.639071424306332e+19,
                      7.639071420196412e+19, 7.639071432523558e+19,
                      7.122948308712312e+19, 7.639071420196412e+19,
                      8.413256058790063e+19, 8.413256071059287e+19,
                      7.122948320807626e+19, 7.639071432523558e+19,
                      8.413256071059287e+19, 9.445502273748306e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{7.122948296615527e+19, 7.122948312744913e+19,
                      7.122948308712188e+19, 7.122948320807332e+19,
                      7.122948312744913e+19, 7.639071424305924e+19,
                      7.639071420196061e+19, 7.639071432523037e+19,
                      7.122948308712188e+19, 7.639071420196061e+19,
                      8.413256058789768e+19, 8.413256071058824e+19,
                      7.122948320807332e+19, 7.639071432523037e+19,
                      8.413256071058824e+19, 9.445502273747671e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{7.122948296615504e+19, 7.122948312745003e+19,
                      7.122948308712251e+19, 7.122948320807479e+19,
                      7.122948312745003e+19, 7.639071424306128e+19,
                      7.639071420196236e+19, 7.639071432523298e+19,
                      7.122948308712251e+19, 7.639071420196236e+19,
                      8.413256058789916e+19, 8.413256071059056e+19,
                      7.122948320807479e+19, 7.639071432523298e+19,
                      8.413256071059056e+19, 9.445502273747989e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.) {
        spec_vals = {{7.122948296615482e+19, 7.122948312745094e+19,
                      7.122948308712312e+19, 7.122948320807626e+19,
                      7.122948312745094e+19, 7.63907142430633e+19,
                      7.63907142019641e+19, 7.639071432523556e+19,
                      7.122948308712312e+19, 7.63907142019641e+19,
                      8.413256058790063e+19, 8.413256071059287e+19,
                      7.122948320807626e+19, 7.639071432523556e+19,
                      8.413256071059287e+19, 9.445502273748306e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{7.122948296615527e+19, 7.122948312744913e+19,
                      7.122948308712188e+19, 7.122948320807332e+19,
                      7.122948312744913e+19, 7.639071424305924e+19,
                      7.639071420196061e+19, 7.639071432523039e+19,
                      7.122948308712188e+19, 7.639071420196061e+19,
                      8.413256058789768e+19, 8.413256071058824e+19,
                      7.122948320807332e+19, 7.639071432523039e+19,
                      8.413256071058824e+19, 9.445502273747671e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 299.5 and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{7.122948296615504e+19, 7.122948312745003e+19,
                      7.122948308712251e+19, 7.122948320807479e+19,
                      7.122948312745003e+19, 7.639071424306128e+19,
                      7.639071420196236e+19, 7.639071432523298e+19,
                      7.122948308712251e+19, 7.639071420196236e+19,
                      8.413256058789916e+19, 8.413256071059056e+19,
                      7.122948320807479e+19, 7.639071432523298e+19,
                      8.413256071059056e+19, 9.445502273747989e+19}};
        not_initialized = false;
      }
      if (local_inertial_coords_x[i] == 300. and
          local_inertial_coords_y[i] == 0.5 and
          local_inertial_coords_z[i] == 0.5) {
        spec_vals = {{7.122948296615481e+19, 7.122948312745094e+19,
                      7.122948308712312e+19, 7.122948320807626e+19,
                      7.122948312745094e+19, 7.639071424306332e+19,
                      7.639071420196412e+19, 7.639071432523558e+19,
                      7.122948308712312e+19, 7.639071420196412e+19,
                      8.413256058790063e+19, 8.413256071059287e+19,
                      7.122948320807626e+19, 7.639071432523558e+19,
                      8.413256071059287e+19, 9.445502273748306e+19}};
        not_initialized = false;
      }
      if (not_initialized) {
        ERROR("Not checking the correct face, coordinates not recognized");
      }
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          spec_bc_dt_v_minus.get(a, b)[i] =
              gsl::at(spec_vals, a * (1 + VolumeDim) + b);
        }
      }
    }

    // Compare values returned by BC action vs those from SpEC
    CHECK_ITERABLE_APPROX(local_bc_dt_v_minus, spec_bc_dt_v_minus);
  }
}
}  // namespace

// Python tests
namespace {
template <size_t VolumeDim>
tnsr::aa<DataVector, VolumeDim, Frame::Inertial> wrapper_func_v_psi(
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
        interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds) {
  std::array<DataVector, 4> char_speed_array{
      {get<0>(char_speeds), get<1>(char_speeds), get<2>(char_speeds),
       get<3>(char_speeds)}};
  auto dt_v_psi =
      make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
          interface_normal_vector, 0.);
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::
      constraint_preserving_bjorhus_corrections_dt_v_psi<VolumeDim, DataVector>(
          make_not_null(&dt_v_psi), interface_normal_vector,
          three_index_constraint, char_speed_array);
  return dt_v_psi;
}

template <size_t VolumeDim>
void test_constraint_preserving_bjorhus_corrections_dt_v_psi(
    const size_t grid_size_each_dimension) noexcept {
  pypp::check_with_random_values<1>(
      &wrapper_func_v_psi<VolumeDim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      "constraint_preserving_bjorhus_corrections_dt_v_psi", {{{-1., 1.}}},
      DataVector(grid_size_each_dimension));
}

template <size_t VolumeDim>
tnsr::iaa<DataVector, VolumeDim, Frame::Inertial> wrapper_func_v_zero(
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
        interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds) {
  std::array<DataVector, 4> char_speed_array{
      get<0>(char_speeds), get<1>(char_speeds), get<2>(char_speeds),
      get<3>(char_speeds)};
  auto dt_v_zero =
      make_with_value<tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>>(
          get<0>(interface_normal_vector), 0.);
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::
      constraint_preserving_bjorhus_corrections_dt_v_zero<VolumeDim,
                                                          DataVector>(
          make_not_null(&dt_v_zero), interface_normal_vector,
          four_index_constraint, char_speed_array);
  return dt_v_zero;
}

template <size_t VolumeDim>
void test_constraint_preserving_bjorhus_corrections_dt_v_zero(
    const size_t grid_size_each_dimension) noexcept {
  pypp::check_with_random_values<1>(
      &wrapper_func_v_zero<VolumeDim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      "constraint_preserving_bjorhus_corrections_dt_v_zero", {{{-1., 1.}}},
      DataVector(grid_size_each_dimension));
}

template <size_t VolumeDim>
tnsr::aa<DataVector, VolumeDim, Frame::Inertial> wrapper_func_cp_v_minus(
    const Scalar<DataVector>& gamma2,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds) noexcept {
  std::array<DataVector, 4> char_speed_array{
      get<0>(char_speeds), get<1>(char_speeds), get<2>(char_speeds),
      get<3>(char_speeds)};
  auto dt_v_minus =
      make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
          get(gamma2), 0.);
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::
      constraint_preserving_bjorhus_corrections_dt_v_minus<VolumeDim,
                                                           DataVector>(
          make_not_null(&dt_v_minus), gamma2, inertial_coords,
          incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
          outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
          char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
          constraint_char_zero_plus, constraint_char_zero_minus,
          char_speed_array);
  return dt_v_minus;
}

template <size_t VolumeDim>
tnsr::aa<DataVector, VolumeDim, Frame::Inertial> wrapper_func_cpp_v_minus(
    const Scalar<DataVector>& gamma2,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::ii<DataVector, VolumeDim, Frame::Inertial>& extrinsic_curvature,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& spacetime_metric,
    const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>&
        inverse_spacetime_metric,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
    const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds) noexcept {
  std::array<DataVector, 4> char_speed_array{
      get<0>(char_speeds), get<1>(char_speeds), get<2>(char_speeds),
      get<3>(char_speeds)};
  auto dt_v_minus =
      make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
          get(gamma2), 0.);
  GeneralizedHarmonic::BoundaryConditions::Bjorhus::
      constraint_preserving_physical_bjorhus_corrections_dt_v_minus<VolumeDim,
                                                                    DataVector>(
          make_not_null(&dt_v_minus), gamma2, inertial_coords,
          unit_interface_normal_one_form, unit_interface_normal_vector,
          spacetime_unit_normal_vector, incoming_null_one_form,
          outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
          projection_ab, projection_Ab, projection_AB, inverse_spatial_metric,
          extrinsic_curvature, spacetime_metric, inverse_spacetime_metric,
          three_index_constraint, char_projected_rhs_dt_v_psi,
          char_projected_rhs_dt_v_minus, constraint_char_zero_plus,
          constraint_char_zero_minus, phi, d_phi, d_pi, char_speed_array);
  return dt_v_minus;
}

template <size_t VolumeDim>
void test_constraint_preserving_bjorhus_corrections_dt_v_minus(
    const size_t grid_size_each_dimension) noexcept {
  pypp::check_with_random_values<1>(
      &wrapper_func_cp_v_minus<VolumeDim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      "constraint_preserving_bjorhus_corrections_dt_v_minus", {{{-1., 1.}}},
      DataVector(grid_size_each_dimension));
}

template <size_t VolumeDim>
void test_constraint_preserving_physical_bjorhus_corrections_dt_v_minus(
    const size_t grid_size_each_dimension) noexcept {
  pypp::check_with_random_values<1>(
      &wrapper_func_cpp_v_minus<VolumeDim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      "constraint_preserving_physical_bjorhus_corrections_dt_v_minus",
      {{{-1., 1.}}}, DataVector(grid_size_each_dimension));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus.VPsi",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;

  // Python tests
  test_constraint_preserving_bjorhus_corrections_dt_v_psi<1>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_psi<2>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_psi<3>(grid_size);

  // Piece-wise tests with SpEC output in 3D
  test_constraint_preserving_bjorhus_v_psi_vs_spec_3d(grid_size);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus.VZero",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  // Piece-wise tests with SpEC output
  const size_t grid_size = 3;

  // Python tests
  test_constraint_preserving_bjorhus_corrections_dt_v_zero<1>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_zero<2>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_zero<3>(grid_size);

  // Piece-wise tests with SpEC output
  test_constraint_preserving_bjorhus_v_zero_vs_spec_3d(grid_size);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus.VMinus",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  const size_t grid_size = 3;

  // Python tests
  test_constraint_preserving_bjorhus_corrections_dt_v_minus<1>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_minus<2>(grid_size);
  test_constraint_preserving_bjorhus_corrections_dt_v_minus<3>(grid_size);
  test_constraint_preserving_physical_bjorhus_corrections_dt_v_minus<1>(
      grid_size);
  test_constraint_preserving_physical_bjorhus_corrections_dt_v_minus<2>(
      grid_size);
  test_constraint_preserving_physical_bjorhus_corrections_dt_v_minus<3>(
      grid_size);

  // Piece-wise tests with SpEC output in 3D
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  const std::array<double, 3> upper_bound{{300., 0.5, 0.5}};

  test_constraint_preserving_physical_bjorhus_v_minus_vs_spec_3d(
      grid_size, lower_bound, upper_bound);
}
