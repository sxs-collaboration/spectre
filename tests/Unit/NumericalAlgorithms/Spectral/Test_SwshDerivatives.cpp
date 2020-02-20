// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"   // IWYU pragma: keep
#include "DataStructures/ComplexModalVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare ComplexDataVector
// IWYU pragma: no_forward_declare ComplexModalVector
// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral {
namespace Swsh {
namespace {

template <size_t Index, int Spin>
struct TestTag : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

// This function verifies that the derivative coefficient routine works as
// expected by comparing the result of transforming, multiplying by the
// derivative operator in SWSH coefficients, then inverse transforming to an
// analytical result for the derivative. This effectively tests a `detail`
// feature, but the intermediate check has been important in catching subtle
// bugs previously.
template <typename DerivativeKind, ComplexRepresentation Representation,
          int Spin>
void test_derivative_via_transforms() noexcept {
  // generate coefficients for the transformation
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{2, 7};
  const size_t l_max = size_distribution(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  ComplexModalVector generated_modes{
      size_of_libsharp_coefficient_vector(l_max) * number_of_radial_points};
  TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&generated_modes), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  // fill the expected collocation point data by evaluating the analytic
  // functions. This is very slow and imprecise (due to factorial division), but
  // comparatively simple to formulate.
  SpinWeighted<ComplexDataVector, Spin> computed_collocation{
      number_of_swsh_collocation_points(l_max) * number_of_radial_points};
  ComplexDataVector expected_derivative_collocation{
      number_of_swsh_collocation_points(l_max) * number_of_radial_points};

  // Fill the collocation values for the original function from the generated
  // mode coefficients
  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin, Representation>(&computed_collocation.data(), generated_modes,
                            l_max, number_of_radial_points,
                            TestHelpers::spin_weighted_spherical_harmonic);

  // Fill the collocation values for the derivative of the function using the
  // analytical form of the derivatives of the spin-weighted harmonics.
  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin, Representation>(
      &expected_derivative_collocation, generated_modes, l_max,
      number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind>);

  auto transform_modes = swsh_transform<Representation>(
      l_max, number_of_radial_points, computed_collocation);

  SpinWeighted<ComplexModalVector,
               Spin + Tags::derivative_spin_weight<DerivativeKind>>
      derivative_modes{size_of_libsharp_coefficient_vector(l_max) *
                       number_of_radial_points};
  // apply the derivative operator to the coefficients
  detail::compute_coefficients_of_derivative<DerivativeKind, Spin>(
      make_not_null(&derivative_modes), make_not_null(&transform_modes), l_max,
      number_of_radial_points);
  ComplexDataVector transform_derivative_collocation =
      inverse_swsh_transform<Representation>(l_max, number_of_radial_points,
                                             derivative_modes)
          .data();

  // approximation needs to be a little loose to consistently accommodate the
  // ratios of factorials in the analytic form
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(transform_derivative_collocation,
                               expected_derivative_collocation,
                               angular_derivative_approx);
}

// This function verifies the operation of the compute_derivatives function for
// processing collections of derivative operations which collect equal spin
// objects during the forward and inverse transforms for transform
// optimization. This calls the utility for two derivatives, `DerivativeKind0`
// and `DerivativeKind1`, each applied to two scalars of spin-weight `Spin0` and
// `Spin1`. The result is compared to analytical forms obtained by multiplying
// the derivatives of the basis functions by the generated coefficients.
template <ComplexRepresentation Representation, int Spin0, int Spin1,
          typename DerivativeKind0, typename DerivativeKind1>
void test_compute_angular_derivatives() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> size_distribution{2, 7};
  const size_t l_max = size_distribution(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  using input_tag_list = tmpl::list<TestTag<0, Spin0>, TestTag<1, Spin1>>;
  using derivative_tag_list =
      tmpl::list<Tags::Derivative<TestTag<0, Spin0>, DerivativeKind0>,
                 Tags::Derivative<TestTag<1, Spin1>, DerivativeKind0>,
                 Tags::Derivative<TestTag<0, Spin0>, DerivativeKind1>,
                 Tags::Derivative<TestTag<1, Spin1>, DerivativeKind1>>;
  using collocation_variables_tag =
      ::Tags::Variables<tmpl::append<input_tag_list, derivative_tag_list>>;
  using coefficients_variables_tag = ::Tags::Variables<db::wrap_tags_in<
      Tags::SwshTransform, tmpl::append<input_tag_list, derivative_tag_list>>>;

  auto box = db::create<
      db::AddSimpleTags<collocation_variables_tag, coefficients_variables_tag,
                        Tags::LMax, Tags::NumberOfRadialPoints>,
      db::AddComputeTags<>>(
      db::item_type<collocation_variables_tag>{
          number_of_radial_points * number_of_swsh_collocation_points(l_max)},
      db::item_type<coefficients_variables_tag>{
          size_of_libsharp_coefficient_vector(l_max) * number_of_radial_points},
      l_max, number_of_radial_points);

  ComplexModalVector expected_modes_spin_0{
      number_of_radial_points * size_of_libsharp_coefficient_vector(l_max)};
  TestHelpers::generate_swsh_modes<Spin0>(
      make_not_null(&expected_modes_spin_0), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  ComplexModalVector expected_modes_spin_1{
      number_of_radial_points * size_of_libsharp_coefficient_vector(l_max)};
  TestHelpers::generate_swsh_modes<Spin1>(
      make_not_null(&expected_modes_spin_1), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  const auto coefficients_to_analytic_collocation = [&l_max](
      const auto computed_collocation,
      const ComplexModalVector& expected_modes) noexcept {
    constexpr int lambda_spin =
        std::decay_t<decltype(*computed_collocation)>::type::spin;
    TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
        lambda_spin, Representation>(
        make_not_null(&get(*computed_collocation).data()), expected_modes,
        l_max, number_of_radial_points,
        TestHelpers::spin_weighted_spherical_harmonic);
  };
  // Put the collocation information derived from the generated modes in the
  // DataBox
  db::mutate<TestTag<0, Spin0>>(make_not_null(&box),
                                coefficients_to_analytic_collocation,
                                expected_modes_spin_0);
  db::mutate<TestTag<1, Spin1>>(make_not_null(&box),
                                coefficients_to_analytic_collocation,
                                expected_modes_spin_1);

  // these could be packed into a variables, but the current test wouldn't be
  // much shorter. If the test is expanded to verify more than four results at a
  // time, using a `Variables` for the expected results is recommended for
  // brevity.
  ComplexDataVector expected_derivative_0_collocation_spin_0{
      number_of_radial_points * number_of_swsh_collocation_points(l_max)};
  ComplexDataVector expected_derivative_0_collocation_spin_1{
      number_of_radial_points * number_of_swsh_collocation_points(l_max)};
  ComplexDataVector expected_derivative_1_collocation_spin_0{
      number_of_radial_points * number_of_swsh_collocation_points(l_max)};
  ComplexDataVector expected_derivative_1_collocation_spin_1{
      number_of_radial_points * number_of_swsh_collocation_points(l_max)};

  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin0, Representation>(
      make_not_null(&expected_derivative_0_collocation_spin_0),
      expected_modes_spin_0, l_max, number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind0>);

  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin1, Representation>(
      make_not_null(&expected_derivative_0_collocation_spin_1),
      expected_modes_spin_1, l_max, number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind0>);

  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin0, Representation>(
      make_not_null(&expected_derivative_1_collocation_spin_0),
      expected_modes_spin_0, l_max, number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind1>);

  TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
      Spin1, Representation>(
      make_not_null(&expected_derivative_1_collocation_spin_1),
      expected_modes_spin_1, l_max, number_of_radial_points,
      TestHelpers::derivative_of_spin_weighted_spherical_harmonic<
          DerivativeKind1>);

  // the actual derivative call
  db::mutate_apply<AngularDerivatives<derivative_tag_list, Representation>>(
      make_not_null(&box));

  // large tolerance due to the necessity of factorial division in the analytic
  // basis function being compared to.
  Approx swsh_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);
  {
    INFO("Check the coefficient data intermediate step");
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_modes_spin_0,
        get(db::get<Tags::SwshTransform<TestTag<0, Spin0>>>(box)).data(),
        swsh_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_modes_spin_1,
        get(db::get<Tags::SwshTransform<TestTag<1, Spin1>>>(box)).data(),
        swsh_approx);
  }
  {
    INFO("Check the collocation derivatives final result");
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_derivative_0_collocation_spin_0,
        get(db::get<Tags::Derivative<TestTag<0, Spin0>, DerivativeKind0>>(box))
            .data(),
        swsh_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_derivative_0_collocation_spin_1,
        get(db::get<Tags::Derivative<TestTag<1, Spin1>, DerivativeKind0>>(box))
            .data(),
        swsh_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_derivative_1_collocation_spin_0,
        get(db::get<Tags::Derivative<TestTag<0, Spin0>, DerivativeKind1>>(box))
            .data(),
        swsh_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        expected_derivative_1_collocation_spin_1,
        get(db::get<Tags::Derivative<TestTag<1, Spin1>, DerivativeKind1>>(box))
            .data(),
        swsh_approx);
  }
  {
    INFO("Check the multiple argument function interface");
    SpinWeighted<ComplexDataVector,
                 Spin0 + Tags::derivative_spin_weight<DerivativeKind0>>
        function_output_0{number_of_radial_points *
                          number_of_swsh_collocation_points(l_max)};
    SpinWeighted<ComplexDataVector,
                 Spin1 + Tags::derivative_spin_weight<DerivativeKind1>>
        function_output_1{number_of_radial_points *
                          number_of_swsh_collocation_points(l_max)};
    angular_derivatives<tmpl::list<DerivativeKind0, DerivativeKind1>,
                        Representation>(
        l_max, number_of_radial_points, make_not_null(&function_output_0),
        make_not_null(&function_output_1), get(db::get<TestTag<0, Spin0>>(box)),
        get(db::get<TestTag<1, Spin1>>(box)));
    CHECK_ITERABLE_CUSTOM_APPROX(expected_derivative_0_collocation_spin_0,
                                 function_output_0.data(), swsh_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(expected_derivative_1_collocation_spin_1,
                                 function_output_1.data(), swsh_approx);
  }
  {
    INFO("Check the single argument function interface");
    auto function_return = angular_derivative<DerivativeKind0, Representation>(
        l_max, number_of_radial_points, get(db::get<TestTag<0, Spin0>>(box)));
    CHECK_ITERABLE_CUSTOM_APPROX(function_return.data(),
                                 expected_derivative_0_collocation_spin_0,
                                 swsh_approx);
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.AngularDerivatives",
                  "[Unit][NumericalAlgorithms]") {
  // we do not test the full set of combinations of derivatives, spins, and
  // slice kinds due to the slow execution time. We test a handful of each spin,
  // each derivative, and of each slice type.
  {
    INFO("Test evaluation of Eth using generated values");
    test_derivative_via_transforms<Tags::Eth,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::Eth,
                                   ComplexRepresentation::RealsThenImags, 0>();
  }
  {
    INFO("Test evaluation of Ethbar using generated values");
    test_derivative_via_transforms<Tags::Ethbar,
                                   ComplexRepresentation::RealsThenImags, -1>();
    test_derivative_via_transforms<Tags::Ethbar,
                                   ComplexRepresentation::Interleaved, 1>();
  }
  {
    INFO("Test evaluation of EthEth using generated values");
    test_derivative_via_transforms<Tags::EthEth,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::EthEth,
                                   ComplexRepresentation::RealsThenImags, 0>();
  }
  {
    INFO("Test evaluation of EthbarEthbar using generated values");
    test_derivative_via_transforms<Tags::EthbarEthbar,
                                   ComplexRepresentation::Interleaved, 0>();
    test_derivative_via_transforms<Tags::EthbarEthbar,
                                   ComplexRepresentation::RealsThenImags, 2>();
  }
  {
    INFO("Test evaluation of EthEthbar using generated values");
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::Interleaved, -2>();
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::RealsThenImags, 0>();
    test_derivative_via_transforms<Tags::EthEthbar,
                                   ComplexRepresentation::Interleaved, 2>();
  }
  {
    INFO("Test evaluation of EthbarEth using generated values");
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::RealsThenImags, -1>();
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::Interleaved, 0>();
    test_derivative_via_transforms<Tags::EthbarEth,
                                   ComplexRepresentation::RealsThenImags, 1>();
  }
  {
    INFO("Test angular_derivatives utility");
    test_compute_angular_derivatives<ComplexRepresentation::Interleaved, -1, 1,
                                     Tags::Eth, Tags::Ethbar>();
    test_compute_angular_derivatives<ComplexRepresentation::RealsThenImags, -1,
                                     1, Tags::Eth, Tags::Ethbar>();
    test_compute_angular_derivatives<ComplexRepresentation::Interleaved, -1, 1,
                                     Tags::EthEthbar, Tags::EthbarEth>();
    test_compute_angular_derivatives<ComplexRepresentation::RealsThenImags, -1,
                                     1, Tags::EthEthbar, Tags::EthbarEth>();
    test_compute_angular_derivatives<ComplexRepresentation::Interleaved, 0, 2,
                                     Tags::Ethbar, Tags::EthbarEthbar>();
    test_compute_angular_derivatives<ComplexRepresentation::RealsThenImags, -2,
                                     0, Tags::Eth, Tags::EthEth>();
  }
}
}  // namespace
}  // namespace Swsh
}  // namespace Spectral
