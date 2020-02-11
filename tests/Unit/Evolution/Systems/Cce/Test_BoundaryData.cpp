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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

void pypp_test_worldtube_computation_steps() noexcept {
  pypp::SetupLocalPythonEnvironment local_python_env{"Evolution/Systems/Cce/"};

  const size_t num_pts = 5;

  pypp::check_with_random_values<1>(
      &cartesian_to_spherical_coordinates_and_jacobians, "BoundaryData",
      {"cartesian_to_angular_coordinates", "cartesian_to_angular_jacobian",
       "cartesian_to_angular_inverse_jacobian"},
      {{{1.0, 5.0}}}, DataVector{num_pts});

  pypp::check_with_random_values<1>(&null_metric_and_derivative, "BoundaryData",
                                    {"du_null_metric", "null_metric"},
                                    {{{1.0, 5.0}}}, DataVector{num_pts});

  pypp::check_with_random_values<1>(&worldtube_normal_and_derivatives,
                                    "BoundaryData",
                                    {"worldtube_normal", "dt_worldtube_normal"},
                                    {{{1.0, 5.0}}}, DataVector{num_pts});

  pypp::check_with_random_values<1>(&null_vector_l_and_derivatives,
                                    "BoundaryData",
                                    {"du_null_vector_l", "null_vector_l"},
                                    {{{1.0, 5.0}}}, DataVector{num_pts});

  pypp::check_with_random_values<1>(
      &dlambda_null_metric_and_inverse, "BoundaryData",
      {"dlambda_null_metric", "inverse_dlambda_null_metric"}, {{{1.0, 5.0}}},
      DataVector{num_pts});

  pypp::check_with_random_values<1>(&beta_worldtube_data, "BoundaryData",
                                    {"bondi_beta_worldtube_data"},
                                    {{{1.0, 5.0}}}, DataVector{num_pts});

  pypp::check_with_random_values<1>(&bondi_u_worldtube_data, "BoundaryData",
                                    {"bondi_u_worldtube_data"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts}, 1.0e-11);

  pypp::check_with_random_values<1>(&bondi_w_worldtube_data, "BoundaryData",
                                    {"bondi_w_worldtube_data"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts}, 1.0e-11);

  pypp::check_with_random_values<1>(&bondi_j_worldtube_data, "BoundaryData",
                                    {"bondi_j_worldtube_data"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});

  pypp::check_with_random_values<1>(
      &dr_bondi_j, "BoundaryData",
      {"dr_bondi_j_worldtube_data", "dr_bondi_j_denominator"}, {{{1.0, 5.0}}},
      DataVector{num_pts});

  pypp::check_with_random_values<1>(&d2lambda_bondi_r, "BoundaryData",
                                    {"d2lambda_bondi_r"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});

  pypp::check_with_random_values<1>(
      &bondi_q_worldtube_data, "BoundaryData",
      {"bondi_q_worldtube_data", "dr_bondi_u_worldtube_data"}, {{{1.0, 5.0}}},
      DataVector{num_pts}, 1.0e-10);

  pypp::check_with_random_values<1>(&bondi_h_worldtube_data, "BoundaryData",
                                    {"bondi_h_worldtube_data"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});

  pypp::check_with_random_values<1>(&du_j_worldtube_data, "BoundaryData",
                                    {"du_j_worldtube_data"}, {{{1.0, 5.0}}},
                                    DataVector{num_pts});
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
  SphericaliCartesianJ cartesian_to_angular_jacobian{number_of_angular_points};
  CartesianiSphericalJ inverse_cartesian_to_angular_jacobian{
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
void test_bondi_r(const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<size_t> l_dist(3, 6);
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  tnsr::iaa<DataVector, 3> expected_phi{number_of_angular_points};
  fill_with_random_values(make_not_null(&expected_phi), gen,
                          make_not_null(&value_dist));
  tnsr::aa<DataVector, 3> expected_psi{number_of_angular_points};
  fill_with_random_values(make_not_null(&expected_psi), gen,
                          make_not_null(&value_dist));
  get<0, 0>(expected_psi) -= 1.0;
  for (size_t a = 1; a < 4; ++a) {
    expected_psi.get(a, a) += 1.0;
  }
  tnsr::aa<DataVector, 3> expected_dt_psi{number_of_angular_points};
  fill_with_random_values(make_not_null(&expected_dt_psi), gen,
                          make_not_null(&value_dist));

  // test bondi_r now that we have an appropriate angular metric
  tnsr::ii<DataVector, 2> angular_psi{number_of_angular_points};
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      angular_psi.get(A, B) = expected_psi.get(A + 2, B + 2);
    }
  }
  tnsr::aa<DataVector, 3, Frame::RadialNull> expected_psi_null_coords{
      number_of_angular_points};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      expected_psi_null_coords.get(a, b) = expected_psi.get(a, b);
    }
  }
  Scalar<SpinWeighted<ComplexDataVector, 0>> local_bondi_r{
      number_of_angular_points};
  bondi_r(make_not_null(&local_bondi_r), expected_psi_null_coords);
  Scalar<SpinWeighted<ComplexDataVector, 0>> expected_bondi_r{
      number_of_angular_points};
  get(expected_bondi_r).data() =
      std::complex<double>(1.0, 0.0) *
      pow(get(determinant_and_inverse(angular_psi).first), 0.25);
  CHECK_ITERABLE_APPROX(get(local_bondi_r).data(),
                        get(expected_bondi_r).data());
}

template <typename Generator>
void test_d_bondi_r_identities(const gsl::not_null<Generator*> gen) noexcept {
  // more resolution needed because we want high precision on the angular
  // derivative check.
  UniformCustomDistribution<size_t> l_dist(8, 12);
  // distribution chosen to be not too far from the scale of the diagnonal
  // elements in the matrix
  UniformCustomDistribution<double> value_dist{0.1, 0.2};
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
  for (size_t a = 0; a < 4; ++a) {
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
  Approx trace_product_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  expected_dlambda_r *= 0.25 * real(get(bondi_r).data());
  CHECK_ITERABLE_CUSTOM_APPROX(expected_dlambda_r, get<1>(d_bondi_r),
                               trace_product_approx);

  // use the trace identity to evaluate the du_r in the contrived case where the
  // du_null metric is proportional to null_metric.
  DataVector expected_du_r{number_of_angular_points, random_scaling * 2.0};
  expected_du_r *= 0.25 * real(get(bondi_r).data());
  CHECK_ITERABLE_CUSTOM_APPROX(expected_du_r, get<0>(d_bondi_r),
                               trace_product_approx);
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

template <typename DataBoxTagList, typename AnalyticSolution>
void dispatch_to_gh_worldtube_computation_from_analytic(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
    const AnalyticSolution& solution, const double extraction_radius,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to
  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(collocation_points)[collocation_point.offset] =
        extraction_radius * sin(collocation_point.theta) *
        cos(collocation_point.phi);
    get<1>(collocation_points)[collocation_point.offset] =
        extraction_radius * sin(collocation_point.theta) *
        sin(collocation_point.phi);
    get<2>(collocation_points)[collocation_point.offset] =
        extraction_radius * cos(collocation_point.theta);
  }

  const auto kerr_schild_variables = solution.variables(
      collocation_points, 0.0, gr::Solutions::KerrSchild::tags<DataVector>{});

  // direct collocation quantities for processing into the GH form of the
  // worldtube function
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(kerr_schild_variables);
  const auto& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(kerr_schild_variables);
  const auto& d_lapse = get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
      kerr_schild_variables);

  const auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
      kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          kerr_schild_variables);
  const auto& dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto psi = gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  create_bondi_boundary_data(box, phi, pi, psi, extraction_radius, l_max);
}

template <typename DataBoxTagList, typename AnalyticSolution>
void dispatch_to_modal_worldtube_computation_from_analytic(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> box,
    const AnalyticSolution& solution, const double extraction_radius,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to
  tnsr::I<DataVector, 3> collocation_coordinates{number_of_angular_points};
  for (const auto& collocation_point :
       Spectral::Swsh::cached_collocation_metadata<
           Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max)) {
    get<0>(collocation_coordinates)[collocation_point.offset] =
        extraction_radius * sin(collocation_point.theta) *
        cos(collocation_point.phi);
    get<1>(collocation_coordinates)[collocation_point.offset] =
        extraction_radius * sin(collocation_point.theta) *
        sin(collocation_point.phi);
    get<2>(collocation_coordinates)[collocation_point.offset] =
        extraction_radius * cos(collocation_point.theta);
  }
  const auto kerr_schild_variables =
      solution.variables(collocation_coordinates, 0.0,
                         gr::Solutions::KerrSchild::tags<DataVector>{});

  // direct collocation quantities for processing into the GH form of the
  // worldtube function
  const Scalar<DataVector>& lapse =
      get<gr::Tags::Lapse<DataVector>>(kerr_schild_variables);
  const Scalar<DataVector>& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(kerr_schild_variables);
  const auto& d_lapse = get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
      kerr_schild_variables);

  const auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
      kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          kerr_schild_variables);
  const auto& dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  Scalar<DataVector> dr_lapse{number_of_angular_points};
  get(dr_lapse) = (get<0>(collocation_coordinates) * get<0>(d_lapse) +
                   get<1>(collocation_coordinates) * get<1>(d_lapse) +
                   get<2>(collocation_coordinates) * get<2>(d_lapse)) /
                  extraction_radius;
  tnsr::I<DataVector, 3> dr_shift{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    dr_shift.get(i) = (get<0>(collocation_coordinates) * d_shift.get(0, i) +
                       get<1>(collocation_coordinates) * d_shift.get(1, i) +
                       get<2>(collocation_coordinates) * d_shift.get(2, i)) /
                      extraction_radius;
  }
  tnsr::ii<DataVector, 3> dr_spatial_metric{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dr_spatial_metric.get(i, j) =
          (get<0>(collocation_coordinates) * d_spatial_metric.get(0, i, j) +
           get<1>(collocation_coordinates) * d_spatial_metric.get(1, i, j) +
           get<2>(collocation_coordinates) * d_spatial_metric.get(2, i, j)) /
          extraction_radius;
    }
  }

  const auto lapse_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(lapse, l_max);
  const auto dt_lapse_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dt_lapse, l_max);
  const auto dr_lapse_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dr_lapse, l_max);

  const auto shift_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(shift, l_max);
  const auto dt_shift_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dt_shift, l_max);
  const auto dr_shift_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dr_shift, l_max);

  const auto spatial_metric_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(spatial_metric, l_max);
  const auto dt_spatial_metric_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dt_spatial_metric, l_max);
  const auto dr_spatial_metric_coefficients =
      TestHelpers::tensor_to_libsharp_coefficients(dr_spatial_metric, l_max);

  create_bondi_boundary_data(
      box, spatial_metric_coefficients, dt_spatial_metric_coefficients,
      dr_spatial_metric_coefficients, shift_coefficients, dt_shift_coefficients,
      dr_shift_coefficients, lapse_coefficients, dt_lapse_coefficients,
      dr_lapse_coefficients, extraction_radius, l_max);
}

// this tests that the method using modal construction of metric components
// and derivatives gives the same answer as the version that just takes the GH
// quantities.
template <typename Generator>
void test_kerr_schild_boundary_consistency(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0 * value_dist(*gen);

  UniformCustomDistribution<size_t> l_dist(12, 18);
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  using boundary_variables_tag = ::Tags::Variables<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>;

  auto gh_boundary_box = db::create<db::AddSimpleTags<boundary_variables_tag>>(
      db::item_type<boundary_variables_tag>{number_of_angular_points});
  auto modal_boundary_box =
      db::create<db::AddSimpleTags<boundary_variables_tag>>(
          db::item_type<boundary_variables_tag>{number_of_angular_points});

  dispatch_to_gh_worldtube_computation_from_analytic(
      make_not_null(&gh_boundary_box), solution, extraction_radius, l_max);
  dispatch_to_modal_worldtube_computation_from_analytic(
      make_not_null(&modal_boundary_box), solution, extraction_radius, l_max);

  // This can be tightened further with higher l_max above. Q, for instance, has
  // aliasing trouble
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&gh_boundary_box, &modal_boundary_box,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(tag::name());
        const auto& test_lhs = db::get<tag>(gh_boundary_box);
        const auto& test_rhs = db::get<tag>(modal_boundary_box);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     angular_derivative_approx);
      });
}

// this tests that both execution pathways in Schwarzschild produce the expected
// Bondi-like scalar quantities.
template <typename Generator>
void test_schwarzschild_solution(const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0 * value_dist(*gen);

  UniformCustomDistribution<size_t> l_dist(10, 14);
  const size_t l_max = l_dist(*gen);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  using boundary_variables_tag = ::Tags::Variables<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>;

  auto gh_boundary_box = db::create<db::AddSimpleTags<boundary_variables_tag>>(
      db::item_type<boundary_variables_tag>{number_of_angular_points});
  auto modal_boundary_box =
      db::create<db::AddSimpleTags<boundary_variables_tag>>(
          db::item_type<boundary_variables_tag>{number_of_angular_points});
  auto expected_box = db::create<db::AddSimpleTags<boundary_variables_tag>>(
      db::item_type<boundary_variables_tag>{number_of_angular_points, 0.0});

  dispatch_to_gh_worldtube_computation_from_analytic(
      make_not_null(&gh_boundary_box), solution, extraction_radius, l_max);
  dispatch_to_modal_worldtube_computation_from_analytic(
      make_not_null(&modal_boundary_box), solution, extraction_radius, l_max);

  db::mutate<Tags::BoundaryValue<Tags::BondiW>,
             Tags::BoundaryValue<Tags::BondiR>>(
      make_not_null(&expected_box),
      [&extraction_radius, &
       mass ](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                  bondi_w,
              const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                  bondi_r) noexcept {
        get(*bondi_r).data() = extraction_radius;
        get(*bondi_w).data() = -2.0 * mass / pow<2>(extraction_radius);
      });

  // This can be tightened further with higher l_max above.
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&gh_boundary_box, &modal_boundary_box, &expected_box,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(tag::name());
        const auto& test_lhs_0 = db::get<tag>(gh_boundary_box);
        const auto& test_rhs_0 = db::get<tag>(expected_box);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs_0, test_rhs_0,
                                     angular_derivative_approx);
        const auto& test_lhs_1 = db::get<tag>(modal_boundary_box);
        const auto& test_rhs_1 = db::get<tag>(expected_box);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs_1, test_rhs_1,
                                     angular_derivative_approx);
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.BoundaryData", "[Unit][Cce]") {
  pypp_test_worldtube_computation_steps();

  MAKE_GENERATOR(gen);
  test_trigonometric_function_identities(make_not_null(&gen));
  test_bondi_r(make_not_null(&gen));
  test_d_bondi_r_identities(make_not_null(&gen));
  test_dyad_identities(make_not_null(&gen));
  test_kerr_schild_boundary_consistency(make_not_null(&gen));
  test_schwarzschild_solution(make_not_null(&gen));
}
}  // namespace Cce
