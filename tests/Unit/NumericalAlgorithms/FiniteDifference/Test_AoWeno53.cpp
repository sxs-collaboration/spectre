// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Exact.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Python.hpp"
#include "NumericalAlgorithms/FiniteDifference/AoWeno.hpp"

namespace {

template <size_t Dim>
void test() {
  const auto recons =
      [](const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
        const double gamma_hi = 0.85;
        const double gamma_lo = 0.999;
        const double epsilon = 1.0e-12;
        fd::reconstruction::aoweno_53<8>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables, gamma_hi, gamma_lo, epsilon);
      };
  const auto recons_neighbor_data = [](const gsl::not_null<DataVector*>
                                           face_data,
                                       const DataVector& volume_data,
                                       const DataVector& neighbor_data,
                                       const Index<Dim>& volume_extents,
                                       const Index<Dim>& ghost_data_extents,
                                       const Direction<Dim>&
                                           direction_to_reconstruct) {
    const double gamma_hi = 0.85;
    const double gamma_lo = 0.999;
    const double epsilon = 1.0e-12;
    if (direction_to_reconstruct.side() == Side::Upper) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Upper, fd::reconstruction::detail::AoWeno53Reconstructor<8>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, gamma_hi, gamma_lo,
          epsilon);
    }
    if (direction_to_reconstruct.side() == Side::Lower) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Lower, fd::reconstruction::detail::AoWeno53Reconstructor<8>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, gamma_hi, gamma_lo,
          epsilon);
    }
  };

  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(2, 8, 5, recons, recons_neighbor_data);
  TestHelpers::fd::reconstruction::test_with_python(
      Index<Dim>{8}, 5, "AoWeno53", "test_aoweno53", recons,
      recons_neighbor_data);

  // Check the 5th order reconstruction is exact
  const auto recons_5th_order_only =
      [](const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
        const double gamma_hi = 1.0;
        const double gamma_lo = 0.999;
        const double epsilon = 1.0e-12;
        fd::reconstruction::aoweno_53<8>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables, gamma_hi, gamma_lo, epsilon);
      };
  const auto recons_neighbor_data_5th_order_only =
      [](const gsl::not_null<DataVector*> face_data,
         const DataVector& volume_data, const DataVector& neighbor_data,
         const Index<Dim>& volume_extents, const Index<Dim>& ghost_data_extents,
         const Direction<Dim>& direction_to_reconstruct) {
        const double gamma_hi = 1.0;
        const double gamma_lo = 0.999;
        const double epsilon = 1.0e-12;
        if (direction_to_reconstruct.side() == Side::Upper) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Upper,
              fd::reconstruction::detail::AoWeno53Reconstructor<8>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, gamma_hi, gamma_lo,
              epsilon);
        }
        if (direction_to_reconstruct.side() == Side::Lower) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Lower,
              fd::reconstruction::detail::AoWeno53Reconstructor<8>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, gamma_hi, gamma_lo,
              epsilon);
        }
      };

  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(4, 8, 5, recons_5th_order_only, recons_neighbor_data_5th_order_only);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.AoWeno53",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");
  test<1>();
  test<2>();
  test<3>();
}
