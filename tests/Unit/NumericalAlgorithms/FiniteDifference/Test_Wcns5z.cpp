// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Exact.hpp"
#include "Helpers/NumericalAlgorithms/FiniteDifference/Python.hpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t NonlinearWeightExponent, size_t Dim>
void test_function_pointers() {
  const auto function_ptrs = fd::reconstruction::wcns5z_function_pointers<Dim>(
      NonlinearWeightExponent);
  CHECK(get<0>(function_ptrs) ==
        &fd::reconstruction::wcns5z<NonlinearWeightExponent, Dim>);
  using function_type =
      void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
               const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
               const double&);
  CHECK(get<1>(function_ptrs) ==
        static_cast<function_type>(
            &fd::reconstruction::reconstruct_neighbor<
                Side::Lower,
                ::fd::reconstruction::detail::Wcns5zReconstructor<
                    NonlinearWeightExponent>,
                Dim>));
  CHECK(get<2>(function_ptrs) ==
        static_cast<function_type>(
            &fd::reconstruction::reconstruct_neighbor<
                Side::Upper,
                ::fd::reconstruction::detail::Wcns5zReconstructor<
                    NonlinearWeightExponent>,
                Dim>));
}

template <size_t Dim>
void test() {
  // test for q=2 case

  const auto recons =
      [](const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
        const double epsilon = 2.0e-16;

        fd::reconstruction::wcns5z<2>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables, epsilon);
      };
  const auto recons_neighbor_data =
      [](const gsl::not_null<DataVector*> face_data,
         const DataVector& volume_data, const DataVector& neighbor_data,
         const Index<Dim>& volume_extents, const Index<Dim>& ghost_data_extents,
         const Direction<Dim>& direction_to_reconstruct) {
        const double epsilon = 2.0e-16;

        if (direction_to_reconstruct.side() == Side::Upper) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Upper, fd::reconstruction::detail::Wcns5zReconstructor<2>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, epsilon);
        }
        if (direction_to_reconstruct.side() == Side::Lower) {
          fd::reconstruction::reconstruct_neighbor<
              Side::Lower, fd::reconstruction::detail::Wcns5zReconstructor<2>>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, epsilon);
        }
      };

  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(2, 5, 5, recons, recons_neighbor_data);

  TestHelpers::fd::reconstruction::test_with_python(
      Index<Dim>{5}, 5, "Wcns5z", "test_wcns5z", recons, recons_neighbor_data);

  test_function_pointers<1, Dim>();
  test_function_pointers<2, Dim>();

  // check for failing case (nonlinear weight exponent = 3)
  CHECK_THROWS_WITH(
      ([]() { ::fd::reconstruction::wcns5z_function_pointers<Dim>(3); })(),
      Catch::Contains("Nonlinear weight exponent should be 1 or 2"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Wcns5z",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");
  test<1>();
  test<2>();
  test<3>();
}
