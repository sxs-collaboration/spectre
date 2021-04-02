// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
DataVector soln(
    const tnsr::I<DataVector, Dim, Frame::Logical>& coords) noexcept {
  DataVector result =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
          1, get<0>(coords));
  for (size_t d = 1; d < Dim; ++d) {
    result += Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
        1, coords.get(d));
  }
  return result;
}

template <size_t Dim>
void test_rdmp_impl(const std::vector<double>& past_max_values,
                    const std::vector<double>& past_min_values,
                    const double rdmp_delta0, const double rdmp_epsilon,
                    const size_t num_pts_1d, const double rescale_dg_by,
                    const double rescale_subcell_by,
                    const bool expected_tci_triggered) noexcept {
  CAPTURE(Dim);
  CAPTURE(num_pts_1d);
  CAPTURE(rdmp_delta0);
  CAPTURE(rdmp_epsilon);
  CAPTURE(expected_tci_triggered);
  CAPTURE(past_max_values);
  CAPTURE(past_min_values);
  const Mesh<Dim> dg_mesh{num_pts_1d, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{2 * num_pts_1d - 1,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::FaceCentered};
  const auto logical_coords = logical_coordinates(dg_mesh);

  Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> dg_vars(
      dg_mesh.number_of_grid_points());
  get(get<Tags::Scalar>(dg_vars)) = soln(logical_coords);
  for (size_t d = 0; d < Dim; ++d) {
    get<Tags::Vector<Dim>>(dg_vars).get(d) = soln(logical_coords);
  }

  Variables<
      tmpl::list<evolution::dg::subcell::Tags::Inactive<Tags::Scalar>,
                 evolution::dg::subcell::Tags::Inactive<Tags::Vector<Dim>>>>
      subcell_vars{evolution::dg::subcell::fd::project(dg_vars, dg_mesh,
                                                       subcell_mesh.extents())};
  subcell_vars *= rescale_subcell_by;
  dg_vars *= rescale_dg_by;

  CHECK(evolution::dg::subcell::rdmp_tci(
            dg_vars, subcell_vars, past_max_values, past_min_values,
            rdmp_delta0, rdmp_epsilon) == expected_tci_triggered);
  // Swap DG and subcell being active/inactive to make sure there are no
  // assumptions about active or inactive being DG or subcell.
  using std::swap;
  swap(dg_vars, subcell_vars);
  CHECK(evolution::dg::subcell::rdmp_tci(
            dg_vars, subcell_vars, past_max_values, past_min_values,
            rdmp_delta0, rdmp_epsilon) == expected_tci_triggered);
}

template <size_t Dim>
void test_rdmp() noexcept {
  const std::vector<double> past_max_values(Dim + 1, static_cast<double>(Dim));
  const std::vector<double> past_min_values(Dim + 1, -static_cast<double>(Dim));

  // We lower the maximum number of 1d points in 3d in order to reduce total
  // test runtime.
  const size_t maximum_number_of_points_1d =
      Dim == 3 ? 7
               : Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t num_pts_1d = 2; num_pts_1d < maximum_number_of_points_1d;
       ++num_pts_1d) {
    test_rdmp_impl<Dim>(past_max_values, past_min_values, 1.0e-4, 1.0e-3,
                        num_pts_1d, 1.0, 1.0, false);
    for (size_t component_index = 0; component_index < Dim + 1;
         ++component_index) {
      CAPTURE(component_index);
      auto local_past_max_values = past_max_values;
      local_past_max_values[component_index] *= 0.5;
      test_rdmp_impl<Dim>(local_past_max_values, past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0, 1.0, true);
      // Rescale subcell values to be tiny, effectively just checking DG
      test_rdmp_impl<Dim>(local_past_max_values, past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0, 1.0e-5, true);
      // Rescale DG values to be tiny, effectively just checking subcell
      test_rdmp_impl<Dim>(local_past_max_values, past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0e-5, 1.0, true);
      // Rescale DG&subcell values to be tiny, so no trigger
      test_rdmp_impl<Dim>(local_past_max_values, past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0e-5, 1.0e-5, false);

      auto local_past_min_values = past_min_values;
      local_past_min_values[component_index] *= 0.5;
      test_rdmp_impl<Dim>(past_max_values, local_past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0, 1.0, true);
      // Rescale subcell values to be tiny, effectively just checking DG
      test_rdmp_impl<Dim>(past_max_values, local_past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0, 1.0e-5, true);
      // Rescale DG values to be tiny, effectively just checking subcell
      test_rdmp_impl<Dim>(past_max_values, local_past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0e-5, 1.0, true);
      // Rescale DG&subcell values to be tiny, so no trigger
      test_rdmp_impl<Dim>(past_max_values, local_past_min_values, 1.0e-4,
                          1.0e-3, num_pts_1d, 1.0e-5, 1.0e-5, false);
    }
  }
}

template <size_t Dim>
void test_rdmp_max_min() {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  const size_t num_pts = 10;
  auto active_vars = make_with_random_values<
      Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);
  auto inactive_vars = make_with_random_values<Variables<
      tmpl::list<evolution::dg::subcell::Tags::Inactive<Tags::Scalar>,
                 evolution::dg::subcell::Tags::Inactive<Tags::Vector<Dim>>>>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);
  get<Dim - 1>(get<evolution::dg::subcell::Tags::Inactive<Tags::Vector<Dim>>>(
      inactive_vars))[num_pts / 2] = 10.0;
  get<Dim - 1>(get<evolution::dg::subcell::Tags::Inactive<Tags::Vector<Dim>>>(
      inactive_vars))[num_pts / 2 + 1] = -10.0;
  {
    // Check that the inactive vars are actually ignored
    const auto [max, min] =
        evolution::dg::subcell::rdmp_max_min(active_vars, inactive_vars, false);
    REQUIRE(max.size() == Dim + 1);
    REQUIRE(min.size() == Dim + 1);
    CHECK(max[Dim] < 10.0);
    CHECK(min[Dim] > -10.0);
  }
  {
    // Check that the inactive vars are included if requested
    const auto [max, min] =
        evolution::dg::subcell::rdmp_max_min(active_vars, inactive_vars, true);
    REQUIRE(max.size() == Dim + 1);
    REQUIRE(min.size() == Dim + 1);
    CHECK(max[Dim] == 10.0);
    CHECK(min[Dim] == -10.0);
  }
  get<Dim - 1>(get<Tags::Vector<Dim>>(active_vars))[num_pts / 2] = 100.0;
  get<Dim - 1>(get<Tags::Vector<Dim>>(active_vars))[num_pts / 2 + 1] = -200.0;
  {
    // Check that the active vars are used for max and min
    const auto [max, min] =
        evolution::dg::subcell::rdmp_max_min(active_vars, inactive_vars, true);
    REQUIRE(max.size() == Dim + 1);
    REQUIRE(min.size() == Dim + 1);
    CHECK(max[Dim] == 100.0);
    CHECK(min[Dim] == -200.0);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tci.Rdmp", "[Evolution][Unit]") {
  test_rdmp<1>();
  test_rdmp<2>();
  test_rdmp<3>();

  test_rdmp_max_min<1>();
  test_rdmp_max_min<2>();
  test_rdmp_max_min<3>();
}
}  // namespace
