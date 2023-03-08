// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

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
          bool Use9thOrder, bool Use7thOrder, bool UseExteriorCell,
          bool ReturnReconstructionOrder>
void test_function_pointers(const FallbackReconstructorType fallback_recons) {
  const auto function_ptrs = fd::reconstruction::
      positivity_preserving_adaptive_order_function_pointers<Dim, false>(
          PositivityPreserving, Use9thOrder, Use7thOrder, fallback_recons);
  CHECK(get<0>(function_ptrs) ==
        static_cast<detail::ppao_recons_type<Dim, false>>(
            &positivity_preserving_adaptive_order<
                FallbackReconstructor, PositivityPreserving, Use9thOrder,
                Use7thOrder, Dim>));
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
                UseExteriorCell, Dim>));
  CHECK(get<2>(function_ptrs) ==
        static_cast<function_type>(
            &reconstruct_neighbor<
                Side::Upper,
                ::fd::reconstruction::detail::
                    PositivityPreservingAdaptiveOrderReconstructor<
                        FallbackReconstructor, PositivityPreserving,
                        Use9thOrder, Use7thOrder>,
                UseExteriorCell, Dim>));
}

template <size_t Dim, class FallbackReconstructor, bool PositivityPreserving,
          bool Use9thOrder, bool Use7thOrder, bool ReturnReconstructionOrder>
void test_impl(const FallbackReconstructorType fallback_recons) {
  CAPTURE(PositivityPreserving);
  CAPTURE(Use9thOrder);
  CAPTURE(Use7thOrder);
  CAPTURE(ReturnReconstructionOrder);
  CAPTURE(fallback_recons);
  using Recons = fd::reconstruction::detail::
      PositivityPreservingAdaptiveOrderReconstructor<FallbackReconstructor,
                                                     PositivityPreserving,
                                                     Use9thOrder, Use7thOrder>;

  const size_t num_points = Recons::stencil_width();

  std::optional<std::array<std::vector<std::uint8_t>, Dim>>
      reconstruction_order_storage{};
  std::optional<std::array<gsl::span<std::uint8_t>, Dim>>
      reconstruction_order{};

  // Non-const because we want different values if we are checking exact
  // reconstruction or matching to python
  double four_to_the_alpha_5 = pow(4.0, 1.0);
  double six_to_the_alpha_7 = pow(6.0, 1.0);
  double eight_to_the_alpha_9 = pow(8.0, 1.0);

  const auto recons =
      [&four_to_the_alpha_5, &six_to_the_alpha_7, &eight_to_the_alpha_9,
       &reconstruction_order, &reconstruction_order_storage](
          const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_upper_side_of_face_vars,
          const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_lower_side_of_face_vars,
          const gsl::span<const double>& volume_vars,
          const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
          const Index<Dim>& volume_extents, const size_t number_of_variables) {
        if constexpr (ReturnReconstructionOrder) {
          reconstruction_order_storage.emplace();
          reconstruction_order.emplace();
          for (size_t i = 0; i < Dim; ++i) {
            auto order_extents = volume_extents;
            order_extents[i] += 2;
            gsl::at(reconstruction_order_storage.value(), i)
                .resize(order_extents.product());
            // Ensure we have reset the values to max so the min calls are fine.
            std::fill_n(
                gsl::at(reconstruction_order_storage.value(), i).begin(),
                order_extents.product(),
                std::numeric_limits<std::uint8_t>::max());
            gsl::at(reconstruction_order.value(), i) = gsl::span<std::uint8_t>{
                gsl::at(reconstruction_order_storage.value(), i).data(),
                gsl::at(reconstruction_order_storage.value(), i).size()};
          }
          fd::reconstruction::positivity_preserving_adaptive_order<
              FallbackReconstructor, PositivityPreserving, Use9thOrder,
              Use7thOrder>(reconstructed_upper_side_of_face_vars,
                           reconstructed_lower_side_of_face_vars,
                           make_not_null(&reconstruction_order), volume_vars,
                           ghost_cell_vars, volume_extents, number_of_variables,
                           four_to_the_alpha_5, six_to_the_alpha_7,
                           eight_to_the_alpha_9);
        } else {
          (void)reconstruction_order, (void)reconstruction_order_storage;
          fd::reconstruction::positivity_preserving_adaptive_order<
              FallbackReconstructor, PositivityPreserving, Use9thOrder,
              Use7thOrder>(reconstructed_upper_side_of_face_vars,
                           reconstructed_lower_side_of_face_vars, volume_vars,
                           ghost_cell_vars, volume_extents, number_of_variables,
                           four_to_the_alpha_5, six_to_the_alpha_7,
                           eight_to_the_alpha_9);
        }
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
    const auto check_recons_order = [&reconstruction_order_storage]() {
      if (not reconstruction_order_storage.has_value()) {
        return;
      }
      for (size_t d = 0; d < Dim; ++d) {
        for (size_t i = 0;
             i < gsl::at(reconstruction_order_storage.value(), d).size(); ++i) {
          CHECK(static_cast<int>(
                    gsl::at(reconstruction_order_storage.value(), d)[i]) ==
                static_cast<int>(Use9thOrder ? 9 : (Use7thOrder ? 7 : 5)));
        }
      }
    };
    check_recons_order();

    const auto recons_neighbor_data_interior_cell =
        [&four_to_the_alpha_5, &six_to_the_alpha_7, &eight_to_the_alpha_9](
            const gsl::not_null<DataVector*> face_data,
            const DataVector& volume_data, const DataVector& neighbor_data,
            const Index<Dim>& volume_extents,
            const Index<Dim>& ghost_data_extents,
            const Direction<Dim>& direction_to_reconstruct) {
          if (direction_to_reconstruct.side() == Side::Upper) {
            fd::reconstruction::reconstruct_neighbor<Side::Upper, Recons,
                                                     false>(
                face_data, volume_data, neighbor_data, volume_extents,
                ghost_data_extents, direction_to_reconstruct,
                four_to_the_alpha_5, six_to_the_alpha_7, eight_to_the_alpha_9);
          }
          if (direction_to_reconstruct.side() == Side::Lower) {
            fd::reconstruction::reconstruct_neighbor<Side::Lower, Recons,
                                                     false>(
                face_data, volume_data, neighbor_data, volume_extents,
                ghost_data_extents, direction_to_reconstruct,
                four_to_the_alpha_5, six_to_the_alpha_7, eight_to_the_alpha_9);
          }
        };
    TestHelpers::fd::reconstruction::test_reconstruction_is_exact_if_in_basis<
        Dim>(Recons::stencil_width() - 1, num_points, Recons::stencil_width(),
             recons, recons_neighbor_data_interior_cell);
    check_recons_order();
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
  for (size_t d = 0; reconstruction_order_storage.has_value() and d < Dim;
       ++d) {
    for (size_t i = 0;
         i < gsl::at(reconstruction_order_storage.value(), d).size(); ++i) {
      CHECK(static_cast<int>(
                gsl::at(reconstruction_order_storage.value(), d)[i]) >= 1);
      CHECK(static_cast<int>(
                gsl::at(reconstruction_order_storage.value(), d)[i]) <=
            static_cast<int>(Use9thOrder ? 9 : (Use7thOrder ? 7 : 5)));
    }
  }

  test_function_pointers<Dim, FallbackReconstructor, PositivityPreserving,
                         Use9thOrder, Use7thOrder, true,
                         ReturnReconstructionOrder>(fallback_recons);
}

template <class FallbackReconstructor, bool ReturnReconstructionOrder>
void test(const FallbackReconstructorType fallback_recons) {
  const auto order_helper = [&fallback_recons](auto use_order_9,
                                               auto use_order_7) {
    test_impl<1, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);
    test_impl<2, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);
    test_impl<3, FallbackReconstructor, true, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);

    test_impl<1, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);
    test_impl<2, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);
    test_impl<3, FallbackReconstructor, false, decltype(use_order_9)::value,
              decltype(use_order_7)::value, ReturnReconstructionOrder>(
        fallback_recons);
  };

  order_helper(std::false_type{}, std::false_type{});
  order_helper(std::false_type{}, std::true_type{});
  order_helper(std::true_type{}, std::false_type{});
  order_helper(std::true_type{}, std::true_type{});
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.FiniteDifference.PositivityPreservingAdaptiveOrder",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/FiniteDifference/");

  test<detail::MinmodReconstructor, true>(FallbackReconstructorType::Minmod);
  test<detail::MinmodReconstructor, false>(FallbackReconstructorType::Minmod);
  test<detail::MonotonisedCentralReconstructor, true>(
      FallbackReconstructorType::MonotonisedCentral);
  test<detail::MonotonisedCentralReconstructor, false>(
      FallbackReconstructorType::MonotonisedCentral);

  CHECK_THROWS_WITH(
      (fd::reconstruction::
           positivity_preserving_adaptive_order_function_pointers<1, false>(
               true, false, false, FallbackReconstructorType::None)),
      Catch::Contains("Can't have None as the low-order reconstructor in "
                      "positivity_preserving_adaptive_order."));
}

}  // namespace fd::reconstruction
