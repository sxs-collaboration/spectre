// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

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

void check_inertial_retarded_time_utilities() noexcept {
  MAKE_GENERATOR(gen);

  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  UniformCustomDistribution<size_t> l_dist(12, 18);
  const size_t l_max = l_dist(gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points = 5;

  auto time_box = db::create<db::AddSimpleTags<
      Tags::LMax, Tags::InertialRetardedTime, Tags::ComplexInertialRetardedTime,
      Tags::EthInertialRetardedTime, Tags::Exp2Beta,
      ::Tags::dt<Tags::InertialRetardedTime>>>(
           l_max,
      Scalar<DataVector>{number_of_angular_points},
      Scalar<SpinWeighted<ComplexDataVector, 0>>{number_of_angular_points},
      Scalar<SpinWeighted<ComplexDataVector, 1>>{number_of_angular_points},
      Scalar<SpinWeighted<ComplexDataVector, 0>>{number_of_angular_points *
                                                 number_of_radial_points},
      Scalar<DataVector>{number_of_angular_points});

  db::mutate<Tags::Exp2Beta>(
      make_not_null(&time_box),
      [&gen, &value_dist ](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              exp_2_beta) noexcept {
        fill_with_random_values(make_not_null(&get(*exp_2_beta).data()),
                                make_not_null(&gen),
                                make_not_null(&value_dist));
      });
  const double random_time = value_dist(gen);

  db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
      make_not_null(&time_box), random_time);

  for (auto val : get(db::get<Tags::InertialRetardedTime>(time_box))) {
    CHECK(val == random_time);
  }
  db::mutate_apply<PreSwshDerivatives<Tags::ComplexInertialRetardedTime>>(
      make_not_null(&time_box));

  const std::complex<double> complex_random_time{random_time, 0.0};
  for (auto val :
       get(db::get<Tags::ComplexInertialRetardedTime>(time_box)).data()) {
    CHECK(val == complex_random_time);
  }

  db::mutate_apply<
      CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>>(
      make_not_null(&time_box));

  for (size_t i = 0; i < number_of_angular_points; ++i) {
    CHECK(get(db::get<::Tags::dt<Tags::InertialRetardedTime>>(time_box))[i] ==
          real(get(db::get<Tags::Exp2Beta>(time_box))
                   .data()[i + number_of_angular_points *
                                   (number_of_radial_points - 1)]));
  }

  const double random_time_delta = 0.1 * value_dist(gen);
  const ComplexDataVector expected_retarded_time_intermediate_value =
      std::complex<double>(1.0, 0.0) *
      get(db::get<::Tags::dt<Tags::InertialRetardedTime>>(time_box));
  auto expected_eth_retarded_time =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1,
          SpinWeighted<ComplexDataVector, 0>(
              expected_retarded_time_intermediate_value));
  expected_eth_retarded_time.data() *= random_time_delta;
  db::mutate<Tags::ComplexInertialRetardedTime>(
      make_not_null(&time_box),
      [&random_time_delta](
          const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
              complex_retarded_time,
          const Scalar<DataVector>& dt_inertial_time) noexcept {
        get(*complex_retarded_time) = std::complex<double>(1.0, 0.0) *
                                      random_time_delta * get(dt_inertial_time);
      },
      db::get<::Tags::dt<Tags::InertialRetardedTime>>(time_box));
  db::mutate_apply<CalculateScriPlusValue<Tags::EthInertialRetardedTime>>(
      make_not_null(&time_box));
  CHECK_ITERABLE_APPROX(expected_eth_retarded_time,
                        get(db::get<Tags::EthInertialRetardedTime>(time_box)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ScriPlusValues",
                  "[Unit][Evolution]") {
  pypp_test_scri_plus_computation_steps();

  check_inertial_retarded_time_utilities();
}
}  // namespace Cce
