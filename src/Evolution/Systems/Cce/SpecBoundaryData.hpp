// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
/*
 * \brief Compute \f$\gamma_{i j}\f$, \f$\gamma^{i j}\f$,
 * \f$\partial_i \gamma_{j k}\f$, and
 * \f$\partial_t g_{i j}\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    gsl::not_null<tnsr::II<DataVector, 3>*> inverse_cartesian_spatial_metric,
    gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    gsl::not_null<Scalar<DataVector>*> radial_correction_factor,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const tnsr::I<DataVector, 3>& unit_cartesian_coords, size_t l_max) noexcept;

/*!
 * \brief Compute \f$\beta^{i}\f$, \f$\partial_i \beta^{j}\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_shift_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& radial_derivative_correction_factor,
    size_t l_max) noexcept;

/*!
 * \brief Compute \f$\alpha\f$, \f$\partial_i \alpha\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function will apply a correction factor associated with a SpEC
 * bug.
 */
void cartesian_lapse_and_derivatives_from_unnormalized_spec_modes(
    gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& radial_derivative_correction_factor,
    size_t l_max) noexcept;

/*!
 * \brief Process the worldtube data from modal metric components and
 * derivatives with incorrectly normalized radial derivatives from an old
 * version of SpEC to desired Bondi quantities, placing the result in the passed
 * \ref DataBoxGroup.
 *
 * \details
 * The mathematics are a bit complicated for all of the coordinate
 * transformations that are necessary to obtain the Bondi gauge quantities.
 * For full mathematical details, see the documentation for functions in
 * `BoundaryData.hpp` and \cite Barkett2019uae \cite Bishop1998uk.
 *
 * This function takes as input the full set of ADM metric data and its radial
 * and time derivatives on a two-dimensional surface of constant \f$r\f$ and
 * \f$t\f$ in numerical coordinates. This data must be provided as spherical
 * harmonic coefficients in the libsharp format. This data is provided in nine
 * `Tensor`s.
 *
 * Sufficient tags to provide full worldtube boundary data at a particular
 * time are set in `bondi_boundary_data`. In particular, the set of tags in
 * `Tags::characteristic_worldtube_boundary_tags` in the provided \ref
 * DataBoxGroup are assigned to the worldtube boundary values associated with
 * the input metric components.
 *
 * The majority of the mathematical transformations are implemented as a set of
 * individual cascaded functions below. The details of the manipulations that
 * are performed to the input data may be found in the individual functions
 * themselves, which are called in the following order:
 * - `trigonometric_functions_on_swsh_collocation()`
 * - `cartesian_to_spherical_coordinates_and_jacobians()`
 * - `cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes()`
 * - `cartesian_shift_and_derivatives_from_unnormalized_spec_modes()`
 * - `cartesian_lapse_and_derivatives_from_unnormalized_spec_modes()`
 * - `GeneralizedHarmonic::phi()`
 * - `gr::time_derivative_of_spacetime_metric`
 * - `gr::spacetime_metric`
 * - `generalized_harmonic_quantities()`
 * - `worldtube_normal_and_derivatives()`
 * - `null_vector_l_and_derivatives()`
 * - `null_metric_and_derivative()`
 * - `dlambda_null_metric_and_inverse()`
 * - `bondi_r()`
 * - `d_bondi_r()`
 * - `dyads()`
 * - `beta_worldtube_data()`
 * - `bondi_u_worldtube_data()`
 * - `bondi_w_worldtube_data()`
 * - `bondi_j_worldtube_data()`
 * - `dr_bondi_j()`
 * - `d2lambda_bondi_r()`
 * - `bondi_q_worldtube_data()`
 * - `bondi_h_worldtube_data()`
 * - `du_j_worldtube_data()`
 */
template <typename DataBoxTagList>
void create_bondi_boundary_data_from_unnormalized_spec_modes(
    const gsl::not_null<db::DataBox<DataBoxTagList>*> bondi_boundary_data,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const double extraction_radius, const size_t l_max) noexcept {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In future, allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>,
      Tags::detail::WorldtubeNormal, ::Tags::dt<Tags::detail::WorldtubeNormal>,
      Tags::detail::NullL, ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>,
      ::Tags::TempScalar<0, DataVector>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};
  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& cartesian_spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& d_cartesian_spatial_metric = get<
      ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      computation_variables);
  auto& interpolation_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          derivative_buffers);
  Scalar<SpinWeighted<ComplexModalVector, 0>> interpolation_modal_buffer{size};
  auto& eth_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 1>>>(
          derivative_buffers);
  auto& radial_correction_factor =
      get<::Tags::TempScalar<0, DataVector>>(computation_variables);
  cartesian_spatial_metric_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_spatial_metric),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&dt_cartesian_spatial_metric),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      make_not_null(&radial_correction_factor), spatial_metric_coefficients,
      dr_spatial_metric_coefficients, dt_spatial_metric_coefficients,
      inverse_cartesian_to_spherical_jacobian, cartesian_coords, l_max);

  auto& cartesian_shift =
      get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& d_cartesian_shift =
      get<::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, ::Frame::Inertial>>(
          computation_variables);
  auto& dt_cartesian_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          computation_variables);

  cartesian_shift_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_shift), make_not_null(&d_cartesian_shift),
      make_not_null(&dt_cartesian_shift),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      shift_coefficients, dr_shift_coefficients, dt_shift_coefficients,
      inverse_cartesian_to_spherical_jacobian, radial_correction_factor, l_max);

  auto& cartesian_lapse =
      get<gr::Tags::Lapse<DataVector>>(computation_variables);
  auto& d_cartesian_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  cartesian_lapse_and_derivatives_from_unnormalized_spec_modes(
      make_not_null(&cartesian_lapse), make_not_null(&d_cartesian_lapse),
      make_not_null(&dt_cartesian_lapse),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      lapse_coefficients, dr_lapse_coefficients, dt_lapse_coefficients,
      inverse_cartesian_to_spherical_jacobian, radial_correction_factor, l_max);

  auto& phi = get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      computation_variables);
  auto& dt_spacetime_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      computation_variables);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  GeneralizedHarmonic::phi(
      make_not_null(&phi), cartesian_lapse, d_cartesian_lapse, cartesian_shift,
      d_cartesian_shift, cartesian_spatial_metric, d_cartesian_spatial_metric);
  gr::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), cartesian_lapse, dt_cartesian_lapse,
      cartesian_shift, dt_cartesian_shift, cartesian_spatial_metric,
      dt_cartesian_spatial_metric);
  gr::spacetime_metric(make_not_null(&spacetime_metric), cartesian_lapse,
                       cartesian_shift, cartesian_spatial_metric);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(
      make_not_null(&du_null_l), make_not_null(&null_l), dt_worldtube_normal,
      dt_cartesian_lapse, dt_spacetime_metric, dt_cartesian_shift,
      cartesian_lapse, spacetime_metric, cartesian_shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  detail::create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}
}  // namespace Cce
