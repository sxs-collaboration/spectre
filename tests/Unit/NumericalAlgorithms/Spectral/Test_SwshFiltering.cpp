// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Spectral::Swsh {
template <int Spin>
void test_angular_filtering() noexcept {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t number_of_radial_points = 2;
  CAPTURE(l_max);
  CAPTURE(Spin);
  // Generate data uniform in r with all angular modes
  SpinWeighted<ComplexModalVector, Spin> generated_modes;
  generated_modes.data() = make_with_random_values<ComplexModalVector>(
      make_not_null(&gen), make_not_null(&coefficient_distribution),
      size_of_libsharp_coefficient_vector(l_max));
  for (const auto mode : cached_coefficients_metadata(l_max)) {
    if (mode.l < static_cast<size_t>(abs(Spin))) {
      generated_modes.data()[mode.transform_of_real_part_offset] = 0.0;
      generated_modes.data()[mode.transform_of_imag_part_offset] = 0.0;
    }
    if (mode.m == 0) {
      generated_modes.data()[mode.transform_of_real_part_offset] =
          real(generated_modes.data()[mode.transform_of_real_part_offset]);
      generated_modes.data()[mode.transform_of_imag_part_offset] =
          real(generated_modes.data()[mode.transform_of_imag_part_offset]);
    }
  }
  auto pre_filter_angular_data =
      inverse_swsh_transform(l_max, 1, generated_modes);
  SpinWeighted<ComplexDataVector, Spin> to_filter{create_vector_of_n_copies(
      pre_filter_angular_data.data(), number_of_radial_points)};

  // remove the top few modes, emulating the filter process
  for (const auto mode : cached_coefficients_metadata(l_max)) {
    if (mode.l > (l_max - 3)) {
      generated_modes.data()[mode.transform_of_real_part_offset] = 0.0;
      generated_modes.data()[mode.transform_of_imag_part_offset] = 0.0;
    }
  }
  const auto expected_post_filter_angular_data =
      inverse_swsh_transform(l_max, 1, generated_modes);
  const SpinWeighted<ComplexDataVector, Spin> expected_post_filter{
      create_vector_of_n_copies(expected_post_filter_angular_data.data(),
                                number_of_radial_points)};

  filter_swsh_volume_quantity(make_not_null(&to_filter), l_max, l_max - 3, 5.0,
                              2);
  Approx angular_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(to_filter.data(), expected_post_filter.data(),
                               angular_approx);

  filter_swsh_boundary_quantity(make_not_null(&pre_filter_angular_data), l_max,
                                l_max - 3);
  CHECK_ITERABLE_CUSTOM_APPROX(pre_filter_angular_data.data(),
                               expected_post_filter_angular_data.data(),
                               angular_approx);
}

template <int Spin>
void test_radial_filtering() noexcept {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<size_t> sdist{2, 5};
  const size_t l_max = sdist(generator);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  const size_t number_of_radial_points = sdist(generator);
  CAPTURE(l_max);
  CAPTURE(Spin);
  auto modes = make_with_random_values<ComplexModalVector>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      number_of_radial_points);
  SpinWeighted<ComplexDataVector, Spin> to_filter{outer_product(
      ComplexDataVector{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
      to_nodal_coefficients(modes,
                            Mesh<1>{{{modes.size()}},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto}))};

  const double alpha = 10.0;
  const size_t half_power = 4;
  for (size_t i = 0; i < modes.size(); ++i) {
    modes[i] *= exp(-alpha * pow(i / static_cast<double>(modes.size() - 1),
                                 2 * half_power));
  }
  const SpinWeighted<ComplexDataVector, Spin> expected_post_filter{
      outer_product(
          ComplexDataVector{
              Spectral::Swsh::number_of_swsh_collocation_points(l_max), 1.0},
          to_nodal_coefficients(modes,
                                Mesh<1>{{{modes.size()}},
                                        Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto}))};

  filter_swsh_volume_quantity(make_not_null(&to_filter), l_max, l_max,
                              alpha, half_power);

  CHECK_ITERABLE_APPROX(to_filter.data(), expected_post_filter.data());
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshFiltering",
                  "[Unit][NumericalAlgorithms]") {
  test_angular_filtering<-1>();
  test_angular_filtering<0>();
  test_angular_filtering<2>();
  test_radial_filtering<0>();
}
}  // namespace Spectral::Swsh
