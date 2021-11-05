// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Exact.hpp"
#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t Degree, size_t Dim>
void test() {
  CAPTURE(Degree);
  CAPTURE(Dim);
  const auto recons =
      [](const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
        fd::reconstruction::unlimited<Degree>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables);
      };
  const auto recons_neighbor_data =
      [](const gsl::not_null<DataVector*> face_data,
         const DataVector& volume_data, const DataVector& neighbor_data,
         const Index<Dim>& volume_extents, const Index<Dim>& ghost_data_extents,
         const Direction<Dim>& direction_to_reconstruct) {
        if (direction_to_reconstruct.side() == Side::Upper) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Upper,
              fd::reconstruction::detail::UnlimitedReconstructor<Degree>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct);
        }
        if (direction_to_reconstruct.side() == Side::Lower) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Lower,
              fd::reconstruction::detail::UnlimitedReconstructor<Degree>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct);
        }
      };
  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(Degree, 8, Degree + 1, recons, recons_neighbor_data);
  // No python test since if they can reconstruct their degree exactly then the
  // coefficients are correct because there are non-linear terms to deal with
  // shocks.
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Unlimited",
                  "[Unit][NumericalAlgorithms]") {
  test<2, 1>();
  test<2, 2>();
  test<2, 3>();
  test<4, 1>();
  test<4, 2>();
  test<4, 3>();
}
