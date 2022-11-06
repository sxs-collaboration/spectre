// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"  //IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Pi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Phi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::VSpacetimeMetric
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::VZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::VMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::VPlus
// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Index, size_t Dim, typename Frame>
Scalar<DataVector> speed_with_index(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& normal) {
  return Scalar<DataVector>{GeneralizedHarmonic::characteristic_speeds(
      gamma_1, lapse, shift, normal, {})[Index]};
}

template <size_t Dim, typename Frame>
Scalar<DataVector> char_speed_upsi_with_moving_mesh(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const tnsr::i<DataVector, Dim, Frame>& normal,
    const tnsr::I<DataVector, Dim, Frame>& mesh_velocity) {
  return Scalar<DataVector>{GeneralizedHarmonic::characteristic_speeds(
      gamma_1, lapse, shift, normal, {mesh_velocity})[0]};
}

template <size_t Dim, typename Frame>
void test_characteristic_speeds() {
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::CharacteristicSpeedsCompute<Dim, Frame>>(
      "CharacteristicSpeeds");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, Dim, Frame>,
                                    "TestFunctions", "char_speed_upsi",
                                    {{{-2.0, 2.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, Dim, Frame>,
                                    "TestFunctions", "char_speed_uzero",
                                    {{{-2.0, 2.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<3, Dim, Frame>,
                                    "TestFunctions", "char_speed_uminus",
                                    {{{-2.0, 2.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, Dim, Frame>,
                                    "TestFunctions", "char_speed_uplus",
                                    {{{-2.0, 2.0}}}, used_for_size);

  pypp::check_with_random_values<1>(
      char_speed_upsi_with_moving_mesh<Dim, Frame>, "TestFunctions",
      "char_speed_upsi_moving_mesh", {{{-2.0, 2.0}}}, used_for_size);
}

// Test return-by-reference GH char speeds by comparing to Kerr-Schild
template <typename Solution>
void test_characteristic_speeds_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
  const auto upsi_speed = -(1. + get(gamma_1)) * get(shift_dot_normal);
  const auto uzero_speed = -get(shift_dot_normal);
  const auto uplus_speed = -get(shift_dot_normal) + get(lapse);
  const auto uminus_speed = -get(shift_dot_normal) - get(lapse);

  // Check that locally computed fields match returned ones
  std::array<DataVector, 4> char_speeds_from_func{};
  GeneralizedHarmonic::
      CharacteristicSpeedsCompute<spatial_dim, Frame::Inertial>::function(
          make_not_null(&char_speeds_from_func), gamma_1, lapse, shift,
          unit_normal_one_form, {});
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
typename Tag::type field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi,
    const tnsr::i<DataVector, Dim, Frame>& normal_one_form) {
  return get<Tag>(GeneralizedHarmonic::characteristic_fields(
      gamma_2, inverse_spatial_metric, spacetime_metric, pi, phi,
      normal_one_form));
}

template <size_t Dim, typename Frame>
void test_characteristic_fields() {
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::CharacteristicFieldsCompute<Dim, Frame>>(
      "CharacteristicFields");
  const DataVector used_for_size(5);
  // VSpacetimeMetric
  pypp::check_with_random_values<1>(
      field_with_tag<GeneralizedHarmonic::Tags::VSpacetimeMetric<Dim, Frame>,
                     Dim, Frame>,
      "TestFunctions", "char_field_upsi", {{{-2., 2.}}}, used_for_size);
  // VZero
  pypp::check_with_random_values<1>(
      field_with_tag<GeneralizedHarmonic::Tags::VZero<Dim, Frame>, Dim, Frame>,
      "TestFunctions", "char_field_uzero", {{{-2., 2.}}}, used_for_size,
      1.e-9);  // last argument loosens tolerance from
               // default of 1.0e-12 to avoid occasional
               // failures of this test, suspected from
               // accumulated roundoff error
  // VPlus
  pypp::check_with_random_values<1>(
      field_with_tag<GeneralizedHarmonic::Tags::VPlus<Dim, Frame>, Dim, Frame>,
      "TestFunctions", "char_field_uplus", {{{-2., 2.}}}, used_for_size,
      1.e-10);  // last argument loosens tolerance from
                // default of 1.0e-12 to avoid occasional
                // failures of this test, suspected from
                // accumulated roundoff error
  // VMinus
  pypp::check_with_random_values<1>(
      field_with_tag<GeneralizedHarmonic::Tags::VMinus<Dim, Frame>, Dim, Frame>,
      "TestFunctions", "char_field_uminus", {{{-2., 2.}}}, used_for_size,
      1.e-10);  // last argument loosens tolerance from
                // default of 1.0e-12 to avoid occasional
                // failures of this test, suspected from
                // accumulated roundoff error
}

// Test return-by-reference GH char fields by comparing to Kerr-Schild
template <typename Solution>
void test_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
  const auto uvars = GeneralizedHarmonic::characteristic_fields(
      gamma_2, inverse_spatial_metric, spacetime_metric, pi, phi,
      unit_normal_one_form);

  const auto& upsi_from_func =
      get<GeneralizedHarmonic::Tags::VSpacetimeMetric<spatial_dim,
                                                      Frame::Inertial>>(uvars);
  const auto& uzero_from_func =
      get<GeneralizedHarmonic::Tags::VZero<spatial_dim, Frame::Inertial>>(
          uvars);
  const auto& uplus_from_func =
      get<GeneralizedHarmonic::Tags::VPlus<spatial_dim, Frame::Inertial>>(
          uvars);
  const auto& uminus_from_func =
      get<GeneralizedHarmonic::Tags::VMinus<spatial_dim, Frame::Inertial>>(
          uvars);

  CHECK_ITERABLE_APPROX(upsi, upsi_from_func);
  CHECK_ITERABLE_APPROX(uzero, uzero_from_func);
  CHECK_ITERABLE_APPROX(uplus, uplus_from_func);
  CHECK_ITERABLE_APPROX(uminus, uminus_from_func);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim, typename Frame>
typename Tag::type evol_field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::aa<DataVector, Dim, Frame>& u_psi,
    const tnsr::iaa<DataVector, Dim, Frame>& u_zero,
    const tnsr::aa<DataVector, Dim, Frame>& u_plus,
    const tnsr::aa<DataVector, Dim, Frame>& u_minus,
    const tnsr::i<DataVector, Dim, Frame>& normal_one_form) {
  return get<Tag>(
      GeneralizedHarmonic::evolved_fields_from_characteristic_fields(
          gamma_2, u_psi, u_zero, u_plus, u_minus, normal_one_form));
}

template <size_t Dim, typename Frame>
void test_evolved_from_characteristic_fields() {
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::EvolvedFieldsFromCharacteristicFieldsCompute<Dim,
                                                                        Frame>>(
      "EvolvedFieldsFromCharacteristicFields");
  const DataVector used_for_size(5);
  // Psi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<gr::Tags::SpacetimeMetric<Dim, Frame>, Dim, Frame>,
      "TestFunctions", "evol_field_psi", {{{-2., 2.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<GeneralizedHarmonic::Tags::Pi<Dim, Frame>, Dim,
                          Frame>,
      "TestFunctions", "evol_field_pi", {{{-2., 2.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<GeneralizedHarmonic::Tags::Phi<Dim, Frame>, Dim,
                          Frame>,
      "TestFunctions", "evol_field_phi", {{{-2., 2.}}}, used_for_size);
}

// Test return-by-reference GH fundamental fields by comparing to Kerr-Schild
template <typename Solution>
void test_evolved_from_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Set up grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
      GeneralizedHarmonic::evolved_fields_from_characteristic_fields(
          gamma_2, upsi, uzero, uplus, uminus, unit_normal_one_form);
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

  const size_t grid_size = 2;
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

namespace {
template <size_t Dim>
void check_max_char_speed(const DataVector& used_for_size) {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);

  // Fraction of times the test can randomly fail
  const double failure_tolerance = 1e-10;
  // Minimum fraction of the claimed result that must be found in some
  // random trial
  const double check_minimum = 0.9;
  // Minimum fraction of the claimed result that can be found in any
  // random trial (generally only important for 1D where the random
  // vector will point along the maximum speed direction)
  const double check_maximum = 1.0 + 1e-12;

  const double trial_failure_rate =
      Dim == 1
          ? 0.1  // correct value is 0.0, but cannot take log of that
          : (Dim == 2 ? 1.0 - 2.0 * acos(check_minimum) / M_PI : check_minimum);

  const size_t trials =
      static_cast<size_t>(log(failure_tolerance) / log(trial_failure_rate)) + 1;

  const auto lapse = TestHelpers::gr::random_lapse(&gen, used_for_size);
  const auto shift = TestHelpers::gr::random_shift<Dim>(&gen, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim>(&gen, used_for_size);
  std::uniform_real_distribution<> gamma_1_dist(-5.0, 5.0);
  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&gamma_1_dist), used_for_size);
  double max_char_speed = std::numeric_limits<double>::signaling_NaN();
  GeneralizedHarmonic::Tags::ComputeLargestCharacteristicSpeed<
      Dim, Frame::Inertial>::function(make_not_null(&max_char_speed), gamma_1,
                                      lapse, shift, spatial_metric);

  double maximum_observed = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < trials; ++i) {
    const auto unit_one_form = raise_or_lower_index(
        random_unit_normal(&gen, spatial_metric), spatial_metric);

    const auto characteristic_speeds =
        GeneralizedHarmonic::characteristic_speeds(gamma_1, lapse, shift,
                                                   unit_one_form, {});
    double max_speed_in_chosen_direction = 0.0;
    for (const auto& speed : characteristic_speeds) {
      max_speed_in_chosen_direction =
          std::max(max_speed_in_chosen_direction, max(abs(speed)));
    }

    CHECK(max_speed_in_chosen_direction <= max_char_speed * check_maximum);
    maximum_observed =
        std::max(maximum_observed, max_speed_in_chosen_direction);
  }
  CHECK(maximum_observed >= check_minimum * max_char_speed);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.MaxCharSpeed",
                  "[Unit][Evolution]") {
  check_max_char_speed<1>(DataVector(5));
  check_max_char_speed<2>(DataVector(5));
  check_max_char_speed<3>(DataVector(5));
}
