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
template <size_t Dim, typename FallbackReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder>
void test_function_pointers(const FallbackReconstructorType fallback_recons) {
  const auto function_ptrs = fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<Dim>(
          PositivityPreserving, Use9thOrder, Use7thOrder, fallback_recons);
  CHECK(get<0>(function_ptrs) ==
        &positivity_preserving_adaptive_order<FallbackReconstructor,
                                              PositivityPreserving, Use9thOrder,
                                              Use7thOrder, Dim>);
  using function_type =
      void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
               const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
               const double&, const double&, const double&);
  CHECK(get<1>(function_ptrs) ==
        static_cast<function_type>(
            &reconstruct_neighbor<
                Side::Lower,
                ::fd::reconstruction::detail::
                    PositivityPreservingAdaptiveOrderReconstructor<
                        FallbackReconstructor, PositivityPreserving,
                        Use9thOrder, Use7thOrder>,
                Dim>));
  CHECK(get<2>(function_ptrs) ==
        static_cast<function_type>(
            &reconstruct_neighbor<
                Side::Upper,
                ::fd::reconstruction::detail::
                    PositivityPreservingAdaptiveOrderReconstructor<
                        FallbackReconstructor, PositivityPreserving,
                        Use9thOrder, Use7thOrder>,
                Dim>));
}

template <size_t Dim, class FallbackReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder>
void test_impl(const FallbackReconstructorType fallback_recons) {
  CAPTURE(PositivityPreserving);
  CAPTURE(Use9thOrder);
  CAPTURE(Use7thOrder);
  CAPTURE(fallback_recons);
  using Recons = fd::reconstruction::detail::
      PositivityPreservingAdaptiveOrderReconstructor<FallbackReconstructor,
                                                     PositivityPreserving,
                                                     Use9thOrder, Use7thOrder>;

  const size_t num_points = Recons::stencil_width();

  // Non-const because we want different values if we are checking exact
  // reconstruction or matching to python
  double four_to_the_alpha_5 = pow(4.0, 1.0);
  double six_to_the_alpha_7 = pow(6.0, 1.0);
  double eight_to_the_alpha_9 = pow(8.0, 1.0);

  const auto recons =
      [&four_to_the_alpha_5, &six_to_the_alpha_7, &eight_to_the_alpha_9](
          const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_upper_side_of_face_vars,
          const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_lower_side_of_face_vars,
          const gsl::span<const double>& volume_vars,
          const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
          const Index<Dim>& volume_extents, const size_t number_of_variables) {
        fd::reconstruction::positivity_preserving_adaptive_order<
            FallbackReconstructor, PositivityPreserving, Use9thOrder,
            Use7thOrder>(reconstructed_upper_side_of_face_vars,
                         reconstructed_lower_side_of_face_vars, volume_vars,
                         ghost_cell_vars, volume_extents, number_of_variables,
                         four_to_the_alpha_5, six_to_the_alpha_7,
                         eight_to_the_alpha_9);
      };
  const auto recons_neighbor_data =
      [&four_to_the_alpha_5, &six_to_the_alpha_7, &eight_to_the_alpha_9](
          const gsl::not_null<DataVector*> face_data,
          const DataVector& volume_data, const DataVector& neighbor_data,
          const Index<Dim>& volume_extents,
          const Index<Dim>& ghost_data_extents,
          const Direction<Dim>& direction_to_reconstruct) {
        if (direction_to_reconstruct.side() == Side::Upper) {
          fd::reconstruction::reconstruct_neighbor<Side::Upper, Recons>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, four_to_the_alpha_5,
              six_to_the_alpha_7, eight_to_the_alpha_9);
        }
        if (direction_to_reconstruct.side() == Side::Lower) {
          fd::reconstruction::reconstruct_neighbor<Side::Lower, Recons>(
              face_data, volume_data, neighbor_data, volume_extents,
              ghost_data_extents, direction_to_reconstruct, four_to_the_alpha_5,
              six_to_the_alpha_7, eight_to_the_alpha_9);
        }
      };

  if constexpr (not PositivityPreserving) {
    // Since the solution is negative, doing positivity preservation drops the
    // order.
    TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
        Dim>(Recons::stencil_width() - 1, num_points, Recons::stencil_width(),
             recons, recons_neighbor_data);
  }

  four_to_the_alpha_5 = pow(4.0, 4.0);
  six_to_the_alpha_7 = pow(6.0, 4.3);
  eight_to_the_alpha_9 = pow(8.0, 4.6);

  const std::string ao_name =
      PositivityPreserving
          ? std::string{"test_positivity_preserving_adaptive_order"}
          : std::string{"test_adaptive_order"};

  if (fallback_recons == FallbackReconstructorType::Minmod) {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{num_points}, Recons::stencil_width(),
        "PositivityPreservingAdaptiveOrder", ao_name + "_with_minmod", recons,
        recons_neighbor_data, Use9thOrder, Use7thOrder, four_to_the_alpha_5,
        six_to_the_alpha_7, eight_to_the_alpha_9);
  } else if (fallback_recons == FallbackReconstructorType::MonotonisedCentral) {
    TestHelpers::fd::reconstruction::test_with_python(
        Index<Dim>{num_points}, Recons::stencil_width(),
        "PositivityPreservingAdaptiveOrder", ao_name + "_with_mc", recons,
        recons_neighbor_data, Use9thOrder, Use7thOrder, four_to_the_alpha_5,
        six_to_the_alpha_7, eight_to_the_alpha_9);
  } else {
    ERROR("Fallback not yet implemented.");
  }

  test_function_pointers<Dim, FallbackReconstructor, PositivityPreserving,
                         Use9thOrder, Use7thOrder>(fallback_recons);
}

template <class FallbackReconstructor>
void test(const FallbackReconstructorType fallback_recons) {
  const auto order_helper = [&fallback_recons](auto use_order_9,
                                               auto use_order_7) {
    test_impl<1, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);
    test_impl<2, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);
    test_impl<3, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);

    test_impl<1, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);
    test_impl<2, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);
    test_impl<3, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value>(fallback_recons);
  };

  order_helper(std::false_type{}, std::false_type{});
  order_helper(std::false_type{}, std::true_type{});
  order_helper(std::true_type{}, std::false_type{});
  order_helper(std::true_type{}, std::true_type{});
}
}  // namespace

// [[Timeout, 10]]
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
              true, false, false, FallbackReconstructorType::None),
      Catch::Contains("Can't have None as the low-order reconstructor in "
                      "positivity_preserving_adaptive_order."));
}

}  // namespace fd::reconstruction
