// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"  //IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Pi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Phi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPsi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPlus
// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Index, size_t Dim, typename Frame>
Scalar<DataVector> compute_speed_with_index(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& normal) {
  return Scalar<DataVector>{
      GeneralizedHarmonic::CharacteristicSpeedsCompute<Dim, Frame>::function(
          gamma_1, lapse, shift, normal)[Index]};
}

template <size_t Dim, typename Frame>
void test_characteristic_speeds() noexcept {
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(compute_speed_with_index<0, Dim, Frame>,
                                    "TestFunctions", "char_speed_upsi",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(compute_speed_with_index<1, Dim, Frame>,
                                    "TestFunctions", "char_speed_uzero",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(compute_speed_with_index<3, Dim, Frame>,
                                    "TestFunctions", "char_speed_uminus",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(compute_speed_with_index<2, Dim, Frame>,
                                    "TestFunctions", "char_speed_uplus",
                                    {{{-10.0, 10.0}}}, used_for_size);
}

// Test return-by-reference GH char speeds by comparing to Kerr-Schild
template <typename Solution>
void test_characteristic_speeds_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& shift = get<gr::Tags::Shift<spatial_dim>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<spatial_dim>>(vars);

  // Get ingredients
  const auto gamma_1 = make_with_value<Scalar<DataVector>>(x, 0.1);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  // Outward 3-normal to the surface on which characteristic fields are needed
  auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(x, 1.);
  const auto norm_of_one_form =
      magnitude(unit_normal_one_form, inverse_spatial_metric);
  for (size_t i = 0; i < spatial_dim; ++i) {
    unit_normal_one_form.get(i) /= get(norm_of_one_form);
  }

  // Get generalized harmonic characteristic speeds locally
  const auto shift_dot_normal = dot_product(shift, unit_normal_one_form);
  const auto& upsi_speed = -(1. + get(gamma_1)) * get(shift_dot_normal);
  const auto& uzero_speed = -get(shift_dot_normal);
  const auto& uplus_speed = -get(shift_dot_normal) + get(lapse);
  const auto& uminus_speed = -get(shift_dot_normal) - get(lapse);

  // Check that locally computed fields match returned ones
  const auto char_speeds_from_func =
      GeneralizedHarmonic::CharacteristicSpeedsCompute<
          spatial_dim, Frame::Inertial>::function(gamma_1, lapse, shift,
                                                  unit_normal_one_form);
  const auto& upsi_speed_from_func = char_speeds_from_func[0];
  const auto& uzero_speed_from_func = char_speeds_from_func[1];
  const auto& uplus_speed_from_func = char_speeds_from_func[2];
  const auto& uminus_speed_from_func = char_speeds_from_func[3];

  CHECK_ITERABLE_APPROX(upsi_speed, upsi_speed_from_func);
  CHECK_ITERABLE_APPROX(uzero_speed, uzero_speed_from_func);
  CHECK_ITERABLE_APPROX(uplus_speed, uplus_speed_from_func);
  CHECK_ITERABLE_APPROX(uminus_speed, uminus_speed_from_func);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim, typename Frame>
typename Tag::type compute_field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& normal_one_form) {
  return get<Tag>(
      GeneralizedHarmonic::CharacteristicFieldsCompute<Dim, Frame>::function(
          gamma_2, inverse_spatial_metric, spacetime_metric, pi, phi,
          normal_one_form));
}

template <size_t Dim, typename Frame>
void test_characteristic_fields() noexcept {
  const DataVector used_for_size(20);
  // UPsi
  pypp::check_with_random_values<1>(
      compute_field_with_tag<GeneralizedHarmonic::Tags::UPsi<Dim, Frame>, Dim,
                             Frame>,
      "TestFunctions", "char_field_upsi", {{{-100., 100.}}}, used_for_size);
  // UZero
  pypp::check_with_random_values<1>(
      compute_field_with_tag<GeneralizedHarmonic::Tags::UZero<Dim, Frame>, Dim,
                             Frame>,
      "TestFunctions", "char_field_uzero", {{{-100., 100.}}}, used_for_size,
      1.e-10);
  // UPlus
  pypp::check_with_random_values<1>(
      compute_field_with_tag<GeneralizedHarmonic::Tags::UPlus<Dim, Frame>, Dim,
                             Frame>,
      "TestFunctions", "char_field_uplus", {{{-100., 100.}}}, used_for_size,
      1.e-11);
  // UMinus
  pypp::check_with_random_values<1>(
      compute_field_with_tag<GeneralizedHarmonic::Tags::UMinus<Dim, Frame>, Dim,
                             Frame>,
      "TestFunctions", "char_field_uminus", {{{-100., 100.}}}, used_for_size,
      1.e-10);
}

// Test return-by-reference GH char fields by comparing to Kerr-Schild
template <typename Solution>
void test_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<spatial_dim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<spatial_dim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const size_t n_pts = x.begin()->size();
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<spatial_dim, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);
  // Outward 3-normal to the surface on which characteristic fields are needed
  auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(x, 1.);
  const auto norm_of_one_form =
      magnitude(unit_normal_one_form, inverse_spatial_metric);
  for (size_t i = 0; i < spatial_dim; ++i) {
    unit_normal_one_form.get(i) /= get(norm_of_one_form);
  }
  const auto normal =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  // Compute characteristic fields locally
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> phi_dot_normal{
      DataVector(n_pts, 0.)};
  // Compute phi_dot_normal_{ab} = n^i \Phi_{iab}
  for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
    for (size_t nu = 0; nu < mu + 1; ++nu) {
      for (size_t i = 0; i < spatial_dim; ++i) {
        phi_dot_normal.get(mu, nu) += normal.get(i) * phi.get(i, mu, nu);
      }
    }
  }
  tnsr::iaa<DataVector, spatial_dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts, 0.)};
  // Compute phi_dot_projection_tensor_{kab} = projection_tensor^i_k \Phi_{kab}
  for (size_t i = 0; i < spatial_dim; ++i) {
    for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
      for (size_t nu = 0; nu < mu + 1; ++nu) {
        phi_dot_projection_tensor.get(i, mu, nu) =
            phi.get(i, mu, nu) -
            unit_normal_one_form.get(i) * phi_dot_normal.get(mu, nu);
      }
    }
  }
  // Eq.(32)-(34) of Lindblom+ (2005)
  const auto& upsi = spacetime_metric;
  const auto& uzero = phi_dot_projection_tensor;
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> uplus{
      DataVector(n_pts, 0.)};
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> uminus{
      DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
    for (size_t nu = 0; nu < mu + 1; ++nu) {
      uplus.get(mu, nu) = pi.get(mu, nu) + phi_dot_normal.get(mu, nu) -
                          get(gamma_2) * spacetime_metric.get(mu, nu);
      uminus.get(mu, nu) = pi.get(mu, nu) - phi_dot_normal.get(mu, nu) -
                           get(gamma_2) * spacetime_metric.get(mu, nu);
    }
  }

  // Check that locally computed fields match returned ones
  const auto uvars = GeneralizedHarmonic::CharacteristicFieldsCompute<
      spatial_dim, Frame::Inertial>::function(gamma_2, inverse_spatial_metric,
                                              spacetime_metric, pi, phi,
                                              unit_normal_one_form);

  const auto& upsi_from_func =
      get<GeneralizedHarmonic::Tags::UPsi<spatial_dim, Frame::Inertial>>(uvars);
  const auto& uzero_from_func =
      get<GeneralizedHarmonic::Tags::UZero<spatial_dim, Frame::Inertial>>(
          uvars);
  const auto& uplus_from_func =
      get<GeneralizedHarmonic::Tags::UPlus<spatial_dim, Frame::Inertial>>(
          uvars);
  const auto& uminus_from_func =
      get<GeneralizedHarmonic::Tags::UMinus<spatial_dim, Frame::Inertial>>(
          uvars);

  CHECK_ITERABLE_APPROX(upsi, upsi_from_func);
  CHECK_ITERABLE_APPROX(uzero, uzero_from_func);
  CHECK_ITERABLE_APPROX(uplus, uplus_from_func);
  CHECK_ITERABLE_APPROX(uminus, uminus_from_func);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim, typename Frame>
typename Tag::type compute_evol_field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& normal_one_form) {
  return get<Tag>(
      GeneralizedHarmonic::EvolvedFieldsFromCharacteristicFieldsCompute<
          Dim, Frame>::function(gamma_2, u_psi, u_zero, u_plus, u_minus,
                                normal_one_form));
}

template <size_t Dim, typename Frame>
void test_evolved_from_characteristic_fields() noexcept {
  const DataVector used_for_size(20);
  // Psi
  pypp::check_with_random_values<1>(
      compute_evol_field_with_tag<gr::Tags::SpacetimeMetric<Dim, Frame>, Dim,
                                  Frame>,
      "TestFunctions", "evol_field_psi", {{{-100., 100.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(
      compute_evol_field_with_tag<GeneralizedHarmonic::Tags::Pi<Dim, Frame>,
                                  Dim, Frame>,
      "TestFunctions", "evol_field_pi", {{{-100., 100.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      compute_evol_field_with_tag<GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
                                  Dim, Frame>,
      "TestFunctions", "evol_field_phi", {{{-100., 100.}}}, used_for_size);
}

// Test return-by-reference GH fundamental fields by comparing to Kerr-Schild
template <typename Solution>
void test_evolved_from_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<spatial_dim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<spatial_dim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const size_t n_pts = x.begin()->size();
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<spatial_dim, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);
  // Outward 3-normal to the surface on which characteristic fields are needed
  auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(x, 1.);
  const auto norm_of_one_form =
      magnitude(unit_normal_one_form, inverse_spatial_metric);
  for (size_t i = 0; i < spatial_dim; ++i) {
    unit_normal_one_form.get(i) /= get(norm_of_one_form);
  }
  const auto normal =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);

  // Fundamental fields (psi, pi, phi) have already been computed locally.
  // Now, check that these locally computed fields match returned ones
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> phi_dot_normal{
      DataVector(n_pts, 0.)};
  // Compute phi_dot_normal_{ab} = n^i \Phi_{iab}
  for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
    for (size_t nu = 0; nu < mu + 1; ++nu) {
      for (size_t i = 0; i < spatial_dim; ++i) {
        phi_dot_normal.get(mu, nu) += normal.get(i) * phi.get(i, mu, nu);
      }
    }
  }
  tnsr::iaa<DataVector, spatial_dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts, 0.)};
  // Compute phi_dot_projection_tensor_{kab} = projection_tensor^i_k \Phi_{kab}
  for (size_t i = 0; i < spatial_dim; ++i) {
    for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
      for (size_t nu = 0; nu < mu + 1; ++nu) {
        phi_dot_projection_tensor.get(i, mu, nu) =
            phi.get(i, mu, nu) -
            unit_normal_one_form.get(i) * phi_dot_normal.get(mu, nu);
      }
    }
  }
  // Eq.(32)-(34) of Lindblom+ (2005)
  const auto& upsi = spacetime_metric;
  const auto& uzero = phi_dot_projection_tensor;
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> uplus{
      DataVector(n_pts, 0.)};
  tnsr::aa<DataVector, spatial_dim, Frame::Inertial> uminus{
      DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < spatial_dim + 1; ++mu) {
    for (size_t nu = 0; nu < mu + 1; ++nu) {
      uplus.get(mu, nu) = pi.get(mu, nu) + phi_dot_normal.get(mu, nu) -
                          get(gamma_2) * spacetime_metric.get(mu, nu);
      uminus.get(mu, nu) = pi.get(mu, nu) - phi_dot_normal.get(mu, nu) -
                           get(gamma_2) * spacetime_metric.get(mu, nu);
    }
  }
  const auto ffields =
      GeneralizedHarmonic::EvolvedFieldsFromCharacteristicFieldsCompute<
          spatial_dim, Frame::Inertial>::function(gamma_2, upsi, uzero, uplus,
                                                  uminus, unit_normal_one_form);
  const auto& psi_from_func =
      get<gr::Tags::SpacetimeMetric<spatial_dim, Frame::Inertial>>(ffields);
  const auto& pi_from_func =
      get<GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>(ffields);
  const auto& phi_from_func =
      get<GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>(
          ffields);

  CHECK_ITERABLE_APPROX(spacetime_metric, psi_from_func);
  CHECK_ITERABLE_APPROX(pi, pi_from_func);
  CHECK_ITERABLE_APPROX(phi, phi_from_func);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  test_characteristic_speeds<1, Frame::Grid>();
  test_characteristic_speeds<2, Frame::Grid>();
  test_characteristic_speeds<3, Frame::Grid>();
  test_characteristic_speeds<1, Frame::Inertial>();
  test_characteristic_speeds<2, Frame::Inertial>();
  test_characteristic_speeds<3, Frame::Inertial>();

  // Test GH characteristic speeds against Kerr Schild
  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};

  test_characteristic_speeds_analytic(solution, grid_size, lower_bound,
                                      upper_bound);

  test_characteristic_fields<1, Frame::Grid>();
  test_characteristic_fields<2, Frame::Grid>();
  test_characteristic_fields<3, Frame::Grid>();
  test_characteristic_fields<1, Frame::Inertial>();
  test_characteristic_fields<2, Frame::Inertial>();
  test_characteristic_fields<3, Frame::Inertial>();

  test_evolved_from_characteristic_fields<1, Frame::Grid>();
  test_evolved_from_characteristic_fields<2, Frame::Grid>();
  test_evolved_from_characteristic_fields<3, Frame::Grid>();
  test_evolved_from_characteristic_fields<1, Frame::Inertial>();
  test_evolved_from_characteristic_fields<2, Frame::Inertial>();
  test_evolved_from_characteristic_fields<3, Frame::Inertial>();

  // Test GH characteristic fields against Kerr Schild
  test_characteristic_fields_analytic(solution, grid_size, lower_bound,
                                      upper_bound);
  test_evolved_from_characteristic_fields_analytic(solution, grid_size,
                                                   lower_bound, upper_bound);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.MaxCharSpeed",
                  "[Unit][Evolution]") {
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            1, Frame::Grid>::apply({{DataVector{1., 4., 3., 2., 5.},
                                     DataVector{2., 8., 10., 6., 4.},
                                     DataVector{1., 7., 3., 2., 5.},
                                     DataVector{7., 3., 4., 2., 1.}}}) == 10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            1, Frame::Grid>::apply({{DataVector{1., 4., 3., 2., 5.},
                                     DataVector{2., 8., 10., 6., 4.},
                                     DataVector{1., 7., 3., -11., 5.},
                                     DataVector{7., 3., 4., 2., 1.}}}) == 11.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            1, Frame::Inertial>::apply({{DataVector{1., 4., 3., 2., 5.},
                                         DataVector{2., 8., 10., 6., 4.},
                                         DataVector{1., 7., 3., 2., 5.},
                                         DataVector{7., 3., 4., 2., 1.}}}) ==
        10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            1, Frame::Inertial>::apply({{DataVector{1., 4., 3., 2., 5.},
                                         DataVector{2., 8., 10., 6., 4.},
                                         DataVector{1., 7., 3., -11., 5.},
                                         DataVector{7., 3., 4., 2., 1.}}}) ==
        11.);

  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            2, Frame::Grid>::apply({{DataVector{1., 4., 3., 2., 5.},
                                     DataVector{2., 8., 7., 6., 4.},
                                     DataVector{1., 10., 3., 2., 5.},
                                     DataVector{7., 3., 4., 2., 1.}}}) == 10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            2, Frame::Grid>::apply({{DataVector{1., 4., 3., 2., 5.},
                                     DataVector{2., 8., 10., 6., 4.},
                                     DataVector{1., 7., 3., 1., 5.},
                                     DataVector{7., 3., 4., 2., -11.}}}) ==
        11.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            2, Frame::Inertial>::apply({{DataVector{1., 4., 3., 2., 5.},
                                         DataVector{2., 8., 7., 6., 4.},
                                         DataVector{1., 10., 3., 2., 5.},
                                         DataVector{7., 3., 4., 2., 1.}}}) ==
        10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            2, Frame::Inertial>::apply({{DataVector{1., 4., 3., 2., 5.},
                                         DataVector{2., 8., 10., 6., 4.},
                                         DataVector{1., 7., 3., 1., 5.},
                                         DataVector{7., 3., 4., 2., -11.}}}) ==
        11.);

  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            3, Frame::Grid>::apply({{DataVector{1., 4., 10., 2., 5.},
                                     DataVector{2., 8., 3., 6., 4.},
                                     DataVector{1., 7., 3., 2., 5.},
                                     DataVector{7., 3., 4., 2., 1.}}}) == 10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            3, Frame::Grid>::apply({{DataVector{1., 4., 3., 2., 5.},
                                     DataVector{2., 8., 10., 6., 4.},
                                     DataVector{1., 7., 3., 2., 5.},
                                     DataVector{7., 3., 4., -11., 1.}}}) ==
        11.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            3, Frame::Inertial>::apply({{DataVector{1., 4., 10., 2., 5.},
                                         DataVector{2., 8., 3., 6., 4.},
                                         DataVector{1., 7., 3., 2., 5.},
                                         DataVector{7., 3., 4., 2., 1.}}}) ==
        10.);
  CHECK(GeneralizedHarmonic::ComputeLargestCharacteristicSpeed<
            3, Frame::Inertial>::apply({{DataVector{1., 4., 3., 2., 5.},
                                         DataVector{2., 8., 10., 6., 4.},
                                         DataVector{1., 7., 3., 2., 5.},
                                         DataVector{7., 3., 4., -11., 1.}}}) ==
        11.);
}
