// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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
