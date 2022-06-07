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

template <size_t Dim, size_t NonlinearWeightExponent, bool UseFallbacktoMc>
void test_function_pointers() {
  const auto function_ptrs = fd::reconstruction::wcns5z_function_pointers<Dim>(
      NonlinearWeightExponent, UseFallbacktoMc);
  CHECK(get<0>(function_ptrs) ==
        &fd::reconstruction::wcns5z<NonlinearWeightExponent, UseFallbacktoMc,
                                    Dim>);
  using function_type =
      void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
               const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
               const double&, const size_t&);
  CHECK(get<1>(function_ptrs) ==
        static_cast<function_type>(
            &fd::reconstruction::reconstruct_neighbor<
                Side::Lower,
                ::fd::reconstruction::detail::Wcns5zReconstructor<
                    NonlinearWeightExponent, UseFallbacktoMc>,
                Dim>));
  CHECK(get<2>(function_ptrs) ==
        static_cast<function_type>(
            &fd::reconstruction::reconstruct_neighbor<
                Side::Upper,
                ::fd::reconstruction::detail::Wcns5zReconstructor<
                    NonlinearWeightExponent, UseFallbacktoMc>,
                Dim>));
}

template <size_t Dim, bool UseFallbacktoMc>
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
        const size_t max_number_of_extrema = 1;
        const double epsilon = 2.0e-16;

        fd::reconstruction::wcns5z<2, UseFallbacktoMc>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables, epsilon,
            max_number_of_extrema);
      };
  const auto recons_neighbor_data = [](const gsl::not_null<DataVector*>
                                           face_data,
                                       const DataVector& volume_data,
                                       const DataVector& neighbor_data,
                                       const Index<Dim>& volume_extents,
                                       const Index<Dim>& ghost_data_extents,
                                       const Direction<Dim>&
                                           direction_to_reconstruct) {
    const size_t max_number_of_extrema = 1;
    const double epsilon = 2.0e-16;

    if (direction_to_reconstruct.side() == Side::Upper) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Upper,
          fd::reconstruction::detail::Wcns5zReconstructor<2, UseFallbacktoMc>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, epsilon,
          max_number_of_extrema);
    }
    if (direction_to_reconstruct.side() == Side::Lower) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Lower,
          fd::reconstruction::detail::Wcns5zReconstructor<2, UseFallbacktoMc>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, epsilon,
          max_number_of_extrema);
    }
  };

  // In general, if there are more than one extrema in any of finite difference
  // stencils reconstruction would be switched to MC which is only exact up to
  // the polynomial of degree 1. However, if we are testing with a single
  // (global) degree 2 polynomial (which is used in
  // `test_reconstruction_is_exact_if_in_basis`) there cannot be more than one
  // extrema. So we are safe to use degree 2 case for our test here in order to
  // confirm that Wcns5z + MC reconstruction does reproduce the accuracy of
  // Wcns5z if not switched to MC.
  TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
      Dim>(2, 5, 5, recons, recons_neighbor_data);

  if (UseFallbacktoMc) {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{5}, 5, "Wcns5z", "test_wcns5z_with_mc", recons,
        recons_neighbor_data);
  } else {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{5}, 5, "Wcns5z", "test_wcns5z", recons,
        recons_neighbor_data);
  }

  test_function_pointers<Dim, 1, UseFallbacktoMc>();
  test_function_pointers<Dim, 2, UseFallbacktoMc>();

  // check for failing case (nonlinear weight exponent = 3)
  CHECK_THROWS_WITH(
      ([]() {
        ::fd::reconstruction::wcns5z_function_pointers<Dim>(3, UseFallbacktoMc);
      })(),
      Catch::Contains("Nonlinear weight exponent should be 1 or 2"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.Wcns5z",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");
  test<1, true>();
  test<1, false>();

  test<2, true>();
  test<2, false>();

  test<3, true>();
  test<3, false>();
}
