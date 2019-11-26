// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

void pypp_test_worldtube_computation_steps() noexcept {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  pypp::check_with_random_values<1>(
      &cartesian_to_spherical_coordinates_and_jacobians, "BoundaryData",
      {"cartesian_to_angular_coordinates", "cartesian_to_angular_jacobian",
       "cartesian_to_angular_inverse_jacobian"},
      {{{0.1, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(&null_metric_and_derivative, "BoundaryData",
                                    {"du_null_metric", "null_metric"},
                                    {{{0.1, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(&worldtube_normal_and_derivatives,
                                    "BoundaryData",
                                    {"worldtube_normal", "dt_worldtube_normal"},
                                    {{{0.1, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(
      &null_vector_l_and_derivatives, "BoundaryData",
      {"du_null_vector_l", "null_vector_l"}, {{{0.1, 10.0}}}, DataVector{1});

  pypp::check_with_random_values<1>(
      &dlambda_null_metric_and_inverse, "BoundaryData",
      {"dlambda_null_metric", "inverse_dlambda_null_metric"}, {{{0.1, 10.0}}},
      DataVector{1});
}

template <typename Generator>
void test_trigonometric_function_identities(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<size_t> l_dist(3, 6);
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Scalar<DataVector> cos_phi{number_of_angular_points};
  Scalar<DataVector> cos_theta{number_of_angular_points};
  Scalar<DataVector> sin_phi{number_of_angular_points};
  Scalar<DataVector> sin_theta{number_of_angular_points};
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);
  const DataVector unity{number_of_angular_points, 1.0};
  {
    INFO("Trigonometric identities");
    CHECK_ITERABLE_APPROX(square(get(cos_phi)) + square(get(sin_phi)), unity);
    CHECK_ITERABLE_APPROX(square(get(cos_theta)) + square(get(sin_theta)),
                          unity);
  }

  tnsr::I<DataVector, 3> cartesian_coords{number_of_angular_points};
  jacobian_tensor cartesian_to_angular_jacobian{number_of_angular_points};
  inverse_jacobian_tensor inverse_cartesian_to_angular_jacobian{
      number_of_angular_points};
  UniformCustomDistribution<double> radius_dist{10.0, 100.0};
  const double extraction_radius = radius_dist(*gen);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_angular_jacobian),
      make_not_null(&inverse_cartesian_to_angular_jacobian), cos_phi, cos_theta,
      sin_phi, sin_theta, extraction_radius);
  DataVector jacobian_identity{number_of_angular_points};
  const DataVector zero{number_of_angular_points, 0.0};
  {
    INFO("Jacobian identity")
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        jacobian_identity = cartesian_to_angular_jacobian.get(i, 0) *
                            inverse_cartesian_to_angular_jacobian.get(0, j);
        for (size_t k = 1; k < 3; ++k) {
          jacobian_identity += cartesian_to_angular_jacobian.get(i, k) *
                               inverse_cartesian_to_angular_jacobian.get(k, j);
        }
        if (i == j) {
          CHECK_ITERABLE_APPROX(jacobian_identity, unity);
        } else {
          CHECK_ITERABLE_APPROX(jacobian_identity, zero);
        }
      }
    }
  }
}

template <typename Generator>
void test_d_bondi_r_identities(const gsl::not_null<Generator*> gen) noexcept {
  // more resolution needed because we want high precision on the angular
  // derivative check.
  UniformCustomDistribution<size_t> l_dist(8, 12);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_r{number_of_angular_points};
  // simple test function so that we can easily work out what the expected
  // angular derivatives should be.
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get(bondi_r).data()[collocation_point.offset] =
        5.0 + sin(collocation_point.theta) * cos(collocation_point.phi);
  }
  tnsr::aa<DataVector, 3, Frame::RadialNull> null_metric{
      number_of_angular_points};
  fill_with_random_values(make_not_null(&null_metric), gen,
                          make_not_null(&value_dist));
  // to make the inverse more well-behaved.
  for(size_t a = 0; a < 4; ++a) {
    null_metric.get(a, a) += 1.0;
  }
  const auto inverse_null_metric = determinant_and_inverse(null_metric).second;
  tnsr::aa<DataVector, 3, Frame::RadialNull> dlambda_null_metric{
      number_of_angular_points};
  fill_with_random_values(make_not_null(&dlambda_null_metric), gen,
                          make_not_null(&value_dist));
  tnsr::aa<DataVector, 3, Frame::RadialNull> du_null_metric{
      number_of_angular_points};
  fill_with_random_values(make_not_null(&du_null_metric), gen,
                          make_not_null(&value_dist));
  // initialize the du_null_metric in a way that allows a trace identity to be
  // used to check the calculation
  tnsr::ii<DataVector, 2> angular_inverse_null_metric{number_of_angular_points};
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      angular_inverse_null_metric.get(A, B) =
          inverse_null_metric.get(A + 2, B + 2);
    }
  }
  const tnsr::II<DataVector, 2> trace_identity_angular_null_metric =
      determinant_and_inverse(angular_inverse_null_metric).second;
  double random_scaling = value_dist(*gen);
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      du_null_metric.get(A + 2, B + 2) =
          random_scaling * trace_identity_angular_null_metric.get(A, B);
    }
  }
  tnsr::a<DataVector, 3, Frame::RadialNull> d_bondi_r{number_of_angular_points};
  Cce::d_bondi_r(make_not_null(&d_bondi_r), bondi_r, dlambda_null_metric,
                 du_null_metric, inverse_null_metric, l_max);
  DataVector expected_dtheta_r{number_of_angular_points};
  DataVector expected_dphi_r{number_of_angular_points};
  for (const auto& collocation_point : collocation) {
    expected_dtheta_r[collocation_point.offset] =
        cos(collocation_point.theta) * cos(collocation_point.phi);
    // note 'pfaffian' derivative with the 1/sin(theta)
    expected_dphi_r[collocation_point.offset] = -sin(collocation_point.phi);
  }
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  CHECK_ITERABLE_CUSTOM_APPROX(get<2>(d_bondi_r), expected_dtheta_r,
                               angular_derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<3>(d_bondi_r), expected_dphi_r,
                               angular_derivative_approx);
  // a slightly different calculational path to improve test robustness
  DataVector expected_dlambda_r{number_of_angular_points, 0.0};
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = 0; B < 2; ++B) {
      expected_dlambda_r += inverse_null_metric.get(A + 2, B + 2) *
                            dlambda_null_metric.get(A + 2, B + 2);
    }
  }
  expected_dlambda_r *= 0.25 * real(get(bondi_r).data());
  CHECK_ITERABLE_APPROX(expected_dlambda_r, get<1>(d_bondi_r));

  // use the trace identity to evaluate the du_r in the contrived case where the
  // du_null metric is proportional to null_metric.
  DataVector expected_du_r{number_of_angular_points, random_scaling * 2.0};
  expected_du_r *= 0.25 * real(get(bondi_r).data());
  CHECK_ITERABLE_CUSTOM_APPROX(expected_du_r, get<0>(d_bondi_r),
                               angular_derivative_approx);
}

template <typename Generator>
void test_dyad_identities(const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<size_t> l_dist(3, 6);
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tnsr::I<ComplexDataVector, 2, Frame::RadialNull> up_dyad{
      number_of_angular_points};
  tnsr::i<ComplexDataVector, 2, Frame::RadialNull> down_dyad{
      number_of_angular_points};
  Cce::dyads(make_not_null(&down_dyad), make_not_null(&up_dyad));
  const ComplexDataVector two{number_of_angular_points,
                              std::complex<double>(2.0, 0.0)};
  const ComplexDataVector zero{number_of_angular_points,
                               std::complex<double>(0.0, 0.0)};

  ComplexDataVector dyad_product{number_of_angular_points, 0.0};
  for (size_t A = 0; A < 2; ++A) {
    dyad_product += up_dyad.get(A) * down_dyad.get(A);
  }
  CHECK_ITERABLE_APPROX(dyad_product, zero);
  dyad_product = 0.0;
  for (size_t A = 0; A < 2; ++A) {
    dyad_product += up_dyad.get(A) * conj(down_dyad.get(A));
  }
  CHECK_ITERABLE_APPROX(dyad_product, two);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.BoundaryData",
                  "[Unit][Evolution]") {
  pypp_test_worldtube_computation_steps();

  MAKE_GENERATOR(gen);
  test_trigonometric_function_identities(make_not_null(&gen));
  test_d_bondi_r_identities(make_not_null(&gen));
  test_dyad_identities(make_not_null(&gen));
}
}  // namespace Cce
