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
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Utilities/Gsl.hpp"

namespace fd::reconstruction {
namespace {
template <size_t Dim, typename FallbackReconstructor, bool PositivityPreserving>
void test_function_pointers(const FallbackReconstructorType fallback_recons) {
  const auto function_ptrs = fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<Dim>(
          PositivityPreserving, fallback_recons);
  CHECK(get<0>(function_ptrs) ==
        &positivity_preserving_adaptive_order<FallbackReconstructor,
                                              PositivityPreserving, Dim>);
  using function_type =
      void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
               const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
               const double&);
  CHECK(get<1>(function_ptrs) ==
        static_cast<function_type>(
            &reconstruct_neighbor<
                Side::Lower,
                ::fd::reconstruction::detail::
                    PositivityPreservingAdaptiveOrderReconstructor<
                        FallbackReconstructor, PositivityPreserving>,
                Dim>));
  CHECK(get<2>(function_ptrs) ==
        static_cast<function_type>(
            &reconstruct_neighbor<
                Side::Upper,
                ::fd::reconstruction::detail::
                    PositivityPreservingAdaptiveOrderReconstructor<
                        FallbackReconstructor, PositivityPreserving>,
                Dim>));
}

template <size_t Dim, class FallbackReconstructor, bool PositivityPreserving>
void test_impl(const FallbackReconstructorType fallback_recons) {
  CAPTURE(PositivityPreserving);
  CAPTURE(fallback_recons);

  const auto recons =
      [](const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_upper_side_of_face_vars,
         const gsl::not_null<std::array<gsl::span<double>, Dim>*>
             reconstructed_lower_side_of_face_vars,
         const gsl::span<const double>& volume_vars,
         const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
         const Index<Dim>& volume_extents, const size_t number_of_variables) {
        const double four_to_the_alpha_5 = pow(4.0, 4.0);

        fd::reconstruction::positivity_preserving_adaptive_order<
            FallbackReconstructor, PositivityPreserving>(
            reconstructed_upper_side_of_face_vars,
            reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
            volume_extents, number_of_variables, four_to_the_alpha_5);
      };
  const auto recons_neighbor_data = [](const gsl::not_null<DataVector*>
                                           face_data,
                                       const DataVector& volume_data,
                                       const DataVector& neighbor_data,
                                       const Index<Dim>& volume_extents,
                                       const Index<Dim>& ghost_data_extents,
                                       const Direction<Dim>&
                                           direction_to_reconstruct) {
    const double four_to_the_alpha_5 = pow(4.0, 4.0);

    if (direction_to_reconstruct.side() == Side::Upper) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Upper, fd::reconstruction::detail::
                           PositivityPreservingAdaptiveOrderReconstructor<
                               FallbackReconstructor, PositivityPreserving>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, four_to_the_alpha_5);
    }
    if (direction_to_reconstruct.side() == Side::Lower) {
      fd::reconstruction::reconstruct_neighbor<
          Side::Lower, fd::reconstruction::detail::
                           PositivityPreservingAdaptiveOrderReconstructor<
                               FallbackReconstructor, PositivityPreserving>>(
          face_data, volume_data, neighbor_data, volume_extents,
          ghost_data_extents, direction_to_reconstruct, four_to_the_alpha_5);
    }
  };

  if constexpr (not PositivityPreserving) {
    // Since the solution is negative, doing positivity preservation drops the
    // order.
    TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
        Dim>(2, 5, 5, recons, recons_neighbor_data);
  }

  const std::string ao_name =
      PositivityPreserving
          ? std::string{"test_positivity_preserving_adaptive_order"}
          : std::string{"test_adaptive_order"};

  if (fallback_recons == FallbackReconstructorType::Minmod) {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{5}, 5, "PositivityPreservingAdaptiveOrder",
        ao_name + "_with_minmod", recons, recons_neighbor_data);
  } else if (fallback_recons == FallbackReconstructorType::MonotonisedCentral) {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{5}, 5, "PositivityPreservingAdaptiveOrder",
        ao_name + "_with_mc", recons, recons_neighbor_data);
  } else {
    ERROR("Fallback not yet implemented.");
  }

  test_function_pointers<Dim, FallbackReconstructor, true>(fallback_recons);
  test_function_pointers<Dim, FallbackReconstructor, false>(fallback_recons);
}

template <class FallbackReconstructor>
void test(const FallbackReconstructorType fallback_recons) {
  test_impl<1, FallbackReconstructor, true>(fallback_recons);
  test_impl<2, FallbackReconstructor, true>(fallback_recons);
  test_impl<3, FallbackReconstructor, true>(fallback_recons);

  test_impl<1, FallbackReconstructor, false>(fallback_recons);
  test_impl<2, FallbackReconstructor, false>(fallback_recons);
  test_impl<3, FallbackReconstructor, false>(fallback_recons);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.FiniteDifference.PositivityPreservingAdaptiveOrder",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");

  test<detail::MinmodReconstructor>(FallbackReconstructorType::Minmod);
  test<detail::MonotonisedCentralReconstructor>(
      FallbackReconstructorType::MonotonisedCentral);

  CHECK_THROWS_WITH(
      fd::reconstruction::
          positivity_preserving_adaptive_order_function_pointers<1>(
              true, FallbackReconstructorType::None),
      Catch::Contains("Can't have None as the low-order reconstructor in "
                      "positivity_preserving_adaptive_order."));
}

}  // namespace fd::reconstruction
