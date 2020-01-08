// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace Cce {

namespace {

template <size_t FixedLMax, size_t FixedNumberOfRadialPoints, typename Mutator,
          typename ReturnType, typename ArgumentTypeList>
struct WrapScriPlusComputationImpl;

template <size_t FixedLMax, size_t FixedNumberOfRadialPoints, typename Mutator,
          typename ReturnType, typename... Arguments>
struct WrapScriPlusComputationImpl<FixedLMax, FixedNumberOfRadialPoints,
                                   Mutator, ReturnType,
                                   tmpl::list<Arguments...>> {
  static void apply(const gsl::not_null<ReturnType*> pass_by_pointer,
                    const Arguments&... arguments) noexcept {
    Mutator::apply(pass_by_pointer, arguments..., FixedLMax,
                   FixedNumberOfRadialPoints);
  }
};

template <size_t FixedLMax, size_t FixedNumberOfRadialPoints, typename Mutator>
using WrapScriPlusComputation = WrapScriPlusComputationImpl<
    FixedLMax, FixedNumberOfRadialPoints, Mutator,
    typename db::item_type<tmpl::front<typename Mutator::return_tags>>,
    tmpl::transform<typename Mutator::tensor_argument_tags,
                    tmpl::bind<db::item_type, tmpl::_1>>>;

void pypp_test_scri_plus_computation_steps() noexcept {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  constexpr size_t l_max = 3;
  constexpr size_t number_of_radial_points = 1;

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<l_max, number_of_radial_points,
                               CalculateScriPlusValue<Tags::News>>::apply,
      "ScriPlusValues", {"news"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<l_max, number_of_radial_points,
                               CalculateScriPlusValue<Tags::TimeIntegral<
                                   Tags::ScriPlus<Tags::Psi4>>>>::apply,
      "ScriPlusValues", {"time_integral_psi_4"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlusFactor<Tags::Psi4>>>::apply,
      "ScriPlusValues", {"constant_factor_psi_4"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi3>>>::apply,
      "ScriPlusValues", {"psi_3"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi2>>>::apply,
      "ScriPlusValues", {"psi_2"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi1>>>::apply,
      "ScriPlusValues", {"psi_1"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlus<Tags::Psi0>>>::apply,
      "ScriPlusValues", {"psi_0"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});

  pypp::check_with_random_values<1>(
      &WrapScriPlusComputation<
          l_max, number_of_radial_points,
          CalculateScriPlusValue<Tags::ScriPlus<Tags::Strain>>>::apply,
      "ScriPlusValues", {"strain"}, {{{0.1, 1.0}}},
      DataVector{Spectral::Swsh::number_of_swsh_collocation_points(l_max)});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ScriPlusValues",
                  "[Unit][Evolution]") {
  pypp_test_scri_plus_computation_steps();
}
}  // namespace Cce
