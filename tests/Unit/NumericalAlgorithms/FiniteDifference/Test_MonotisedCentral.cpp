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
#include "NumericalAlgorithms/FiniteDifference/MonotisedCentral.hpp"

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
        fd::reconstruction::monotised_central(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables);
      };
  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(1, 4, 3, recons);
  TestHelpers::fd::reconstruction::test_with_python(
      Index<Dim>{4}, 3, "MonotisedCentral", "test_monotised_central", recons);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.MonotisedCentral",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");
  test<1>();
  test<2>();
  test<3>();
}
