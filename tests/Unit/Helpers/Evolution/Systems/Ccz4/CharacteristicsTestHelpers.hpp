// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"

namespace TestHelpers::Ccz4::fd::detail {
static constexpr size_t Dim = ::Ccz4::fd::System::volume_dim;

template <typename DataType>
typename std::array<DataType, 16> compute_expected_characteristic_speeds(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Frame::Inertial>& shift,
    const Scalar<DataType>& conformal_factor, const double f,
    const tnsr::i<DataType, Dim, Frame::Inertial>& unit_normal_one_form) {
  constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  std::array<DataType, 16> expected_char_speeds{};
  // tensor sector characteristic speeds
  const DataVector shift_n = get(dot_product(shift, unit_normal_one_form));
  expected_char_speeds[0] = -1.0 * shift_n + get(lapse);  // u_tensor_plus
  expected_char_speeds[1] = -1.0 * shift_n - get(lapse);  // u_tensor_minus

  // vector sector characteristic speeds
  expected_char_speeds[2] =
      shifting_shift ? -1.0 * shift_n : 0.0 * shift_n;    // u_vector1_zero
  expected_char_speeds[3] = -1.0 * shift_n + get(lapse);  // u_vector2_plus
  expected_char_speeds[4] = -1.0 * shift_n - get(lapse);  // u_vector2_minus
  const auto sqrt_f_over_phi = sqrt(f) / get(conformal_factor);
  if (shifting_shift) {
    // u_vector3_plus
    expected_char_speeds[5] = -1.0 * shift_n + sqrt_f_over_phi;
    // u_vector3_minus
    expected_char_speeds[6] = -1.0 * shift_n - sqrt_f_over_phi;
  } else {
    // u_vector3_plus
    expected_char_speeds[5] =
        -0.5 * shift_n +
        0.5 * sqrt(4.0 * f + pow<2>(shift_n * get(conformal_factor))) /
            get(conformal_factor);
    // u_vector3_minus
    expected_char_speeds[6] =
        -0.5 * shift_n -
        0.5 * sqrt(4.0 * f + pow<2>(shift_n * get(conformal_factor))) /
            get(conformal_factor);
  }

  // scalar sector characteristic speeds
  expected_char_speeds[7] =
      shifting_shift ? -1.0 * shift_n : 0.0 * shift_n;     // u_scalar1_zero
  expected_char_speeds[8] = -1.0 * shift_n + get(lapse);   // u_scalar2_plus
  expected_char_speeds[9] = -1.0 * shift_n - get(lapse);   // u_scalar2_minus
  expected_char_speeds[10] = -1.0 * shift_n + get(lapse);  // u_scalar3_plus
  expected_char_speeds[11] = -1.0 * shift_n - get(lapse);  // u_scalar3_minus
  expected_char_speeds[12] =
      -1.0 * shift_n + sqrt(2.0 * get(lapse));  // u_scalar4_plus
  expected_char_speeds[13] =
      -1.0 * shift_n - sqrt(2.0 * get(lapse));  // u_scalar4_minus
  if (shifting_shift) {
    const auto v_s = (2.0) / sqrt(3.0) * sqrt_f_over_phi;
    // u_scalar5_plus
    expected_char_speeds[14] = -1.0 * shift_n + v_s;
    // u_scalar5_minus
    expected_char_speeds[15] = -1.0 * shift_n - v_s;
  } else {
    // u_scalar5_plus
    expected_char_speeds[14] =
        -0.5 * shift_n +
        sqrt(48.0 * f + 9 * pow<2>(shift_n * get(conformal_factor))) /
            (6.0 * get(conformal_factor));
    // u_scalar5_minus
    expected_char_speeds[15] =
        -0.5 * shift_n -
        sqrt(48.0 * f + 9 * pow<2>(shift_n * get(conformal_factor))) /
            (6.0 * get(conformal_factor));
  }

  return expected_char_speeds;
}

template <typename DataType>
typename ::Ccz4::fd::Tags::CharacteristicFields<DataType, Dim,
                                                Frame::Inertial>::type
compute_expected_characteristic_fields(
    const tnsr::i<DataType, Dim, Frame::Inertial>& unit_normal_one_form,
    const tnsr::ii<DataType, Dim, Frame::Inertial>& conformal_spatial_metric,
    const Scalar<DataType>& conformal_factor, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Frame::Inertial>& shift,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::ii<DataType, Dim, Frame::Inertial>& a_tilde,
    const Scalar<DataType>& theta,
    const tnsr::I<DataType, Dim, Frame::Inertial>& gamma_hat,
    const tnsr::I<DataType, Dim, Frame::Inertial>& auxiliary_field_b,
    const tnsr::ijj<DataType, Dim, Frame::Inertial>& d_conformal_spatial_metric,
    const tnsr::i<DataType, Dim, Frame::Inertial>& d_conformal_factor,
    const tnsr::i<DataType, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataType, Dim, Frame::Inertial>& d_shift, const double f) {
  const DataVector shift_n = get(dot_product(shift, unit_normal_one_form));
  constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  typename ::Ccz4::fd::Tags::CharacteristicFields<
      DataType, Dim, Frame::Inertial>::type expected_char_fields{
      get(lapse).size()};

  tnsr::ii<DataType, Dim, Frame::Inertial> spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto q_dd =
      gr::transverse_projection_operator(spatial_metric, unit_normal_one_form);
  tnsr::I<DataType, Dim, Frame::Inertial> unit_normal_vector{};
  ::tenex::evaluate<ti::I>(
      make_not_null(&unit_normal_vector),
      inverse_spatial_metric(ti::I, ti::J) * unit_normal_one_form(ti::j));
  const auto q_Ud = gr::transverse_projection_operator(unit_normal_vector,
                                                       unit_normal_one_form);
  const auto q_UU = gr::transverse_projection_operator(inverse_spatial_metric,
                                                       unit_normal_vector);
  tnsr::ijKL<DataType, Dim, Frame::Inertial> tt_projector{};
  ::tenex::evaluate<ti::i, ti::j, ti::K, ti::L>(
      make_not_null(&tt_projector),
      q_Ud(ti::K, ti::i) * q_Ud(ti::L, ti::j) -
          0.5 * q_UU(ti::K, ti::L) * q_dd(ti::i, ti::j));

  // tensor sector
  auto& u_tnsr_plus =
      get<::Ccz4::fd::Tags::UTensorPlus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&u_tnsr_plus),
      tt_projector(ti::i, ti::j, ti::K, ti::L) *
          (a_tilde(ti::k, ti::l) +
           0.5 * unit_normal_vector(ti::M) *
               d_conformal_spatial_metric(ti::m, ti::k, ti::l)));
  auto& u_tnsr_minus =
      get<::Ccz4::fd::Tags::UTensorMinus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&u_tnsr_minus),
      tt_projector(ti::i, ti::j, ti::K, ti::L) *
          (a_tilde(ti::k, ti::l) -
           0.5 * unit_normal_vector(ti::M) *
               d_conformal_spatial_metric(ti::m, ti::k, ti::l)));

  // vector sector
  auto& u_vector1_zero =
      get<::Ccz4::fd::Tags::UVector1Zero<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  ::tenex::evaluate<ti::i>(make_not_null(&u_vector1_zero),
                           q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) -
                               q_dd(ti::i, ti::j) * gamma_hat(ti::J));
  auto& u_vector2_plus =
      get<::Ccz4::fd::Tags::UVector2Plus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  ::tenex::evaluate<ti::i>(
      make_not_null(&u_vector2_plus),
      q_dd(ti::i, ti::j) * gamma_hat(ti::J) -
          unit_normal_vector(ti::J) * q_Ud(ti::K, ti::i) /
              (conformal_factor() * conformal_factor() * conformal_factor() *
               conformal_factor()) *
              (unit_normal_vector(ti::L) *
                   d_conformal_spatial_metric(ti::l, ti::k, ti::j) +
               2.0 * a_tilde(ti::k, ti::j)));
  auto& u_vector2_minus =
      get<::Ccz4::fd::Tags::UVector2Minus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  ::tenex::evaluate<ti::i>(
      make_not_null(&u_vector2_minus),
      q_dd(ti::i, ti::j) * gamma_hat(ti::J) -
          unit_normal_vector(ti::J) * q_Ud(ti::K, ti::i) /
              (conformal_factor() * conformal_factor() * conformal_factor() *
               conformal_factor()) *
              (unit_normal_vector(ti::L) *
                   d_conformal_spatial_metric(ti::l, ti::k, ti::j) -
               2.0 * a_tilde(ti::k, ti::j)));
  auto& u_vector3_plus =
      get<::Ccz4::fd::Tags::UVector3Plus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  auto& u_vector3_minus =
      get<::Ccz4::fd::Tags::UVector3Minus<DataType, Dim, Frame::Inertial>>(
          expected_char_fields);
  if (shifting_shift) {
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_plus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) -
            1.0 / (sqrt(f) * conformal_factor()) * unit_normal_vector(ti::J) *
                q_dd(ti::i, ti::k) * d_shift(ti::j, ti::K));
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_minus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            1.0 / (sqrt(f) * conformal_factor()) * unit_normal_vector(ti::J) *
                q_dd(ti::i, ti::k) * d_shift(ti::j, ti::K));
  } else {
    Scalar<DataType> v_beta_plus{};
    Scalar<DataType> v_beta_minus{};
    get(v_beta_plus) =
        (shift_n - sqrt(4.0 * f + pow<2>(shift_n * get(conformal_factor))) /
                       get(conformal_factor)) /
        (2.0 * f);
    get(v_beta_minus) =
        (shift_n + sqrt(4.0 * f + pow<2>(shift_n * get(conformal_factor))) /
                       get(conformal_factor)) /
        (2.0 * f);
    Scalar<DataType> v_gamma_plus{};
    Scalar<DataType> v_gamma_minus{};
    get(v_gamma_plus) =
        shift_n * pow<2>(get(conformal_factor)) * get(v_beta_plus);
    get(v_gamma_minus) =
        shift_n * pow<2>(get(conformal_factor)) * get(v_beta_minus);
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_plus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            v_gamma_plus() * q_dd(ti::i, ti::j) * gamma_hat(ti::J) +
            v_beta_plus() * unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) *
                d_shift(ti::j, ti::K));
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_minus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            v_gamma_minus() * q_dd(ti::i, ti::j) * gamma_hat(ti::J) +
            v_beta_minus() * unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) *
                d_shift(ti::j, ti::K));
  }

  // scalar sector
  auto& u_scalar1_zero =
      get<::Ccz4::fd::Tags::UScalar1Zero<DataType>>(expected_char_fields);
  ::tenex::evaluate(make_not_null(&u_scalar1_zero),
                    unit_normal_one_form(ti::i) *
                        (auxiliary_field_b(ti::I) - gamma_hat(ti::I)));
  auto& u_scalar2_plus =
      get<::Ccz4::fd::Tags::UScalar2Plus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar2_plus),
      unit_normal_vector(ti::I) * unit_normal_vector(ti::J) *
              (a_tilde(ti::i, ti::j) +
               0.5 * unit_normal_vector(ti::K) *
                   d_conformal_spatial_metric(ti::k, ti::i, ti::j)) +
          2.0 * conformal_factor() * unit_normal_vector(ti::K) *
              d_conformal_factor(ti::k) -
          2.0 / 3.0 * conformal_factor() * conformal_factor() *
              trace_extrinsic_curvature());
  auto& u_scalar2_minus =
      get<::Ccz4::fd::Tags::UScalar2Minus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar2_minus),
      unit_normal_vector(ti::I) * unit_normal_vector(ti::J) *
              (a_tilde(ti::i, ti::j) -
               0.5 * unit_normal_vector(ti::K) *
                   d_conformal_spatial_metric(ti::k, ti::i, ti::j)) -
          2.0 * conformal_factor() * unit_normal_vector(ti::K) *
              d_conformal_factor(ti::k) -
          2.0 / 3.0 * conformal_factor() * conformal_factor() *
              trace_extrinsic_curvature());
  auto& u_scalar3_plus =
      get<::Ccz4::fd::Tags::UScalar3Plus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar3_plus),
      unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
          2.0 / (conformal_factor() * conformal_factor()) *
              (2.0 * unit_normal_vector(ti::I) * d_conformal_factor(ti::i) /
                   conformal_factor() -
               theta()));
  auto& u_scalar3_minus =
      get<::Ccz4::fd::Tags::UScalar3Minus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar3_minus),
      unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
          2.0 / (conformal_factor() * conformal_factor()) *
              (2.0 * unit_normal_vector(ti::I) * d_conformal_factor(ti::i) /
                   conformal_factor() +
               theta()));
  auto& u_scalar4_plus =
      get<::Ccz4::fd::Tags::UScalar4Plus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar4_plus),
      unit_normal_vector(ti::I) * d_lapse(ti::i) +
          sqrt(2.0 * lapse()) * (trace_extrinsic_curvature() - 2.0 * theta()));
  auto& u_scalar4_minus =
      get<::Ccz4::fd::Tags::UScalar4Minus<DataType>>(expected_char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar4_minus),
      unit_normal_vector(ti::I) * d_lapse(ti::i) -
          sqrt(2.0 * lapse()) * (trace_extrinsic_curvature() - 2.0 * theta()));

  auto& u_scalar5_plus =
      get<::Ccz4::fd::Tags::UScalar5Plus<DataType>>(expected_char_fields);
  auto& u_scalar5_minus =
      get<::Ccz4::fd::Tags::UScalar5Minus<DataType>>(expected_char_fields);
  if (shifting_shift) {
    const DataType denom_1 =
        -4.0 * f + 3.0 * pow<2>(get(lapse) * get(conformal_factor));
    const DataType denom_2 =
        -2.0 * f + 3.0 * get(lapse) * pow<2>(get(conformal_factor));
    Scalar<DataType> c_phi{};
    Scalar<DataType> c_gamma{};
    Scalar<DataType> c_alpha{};
    Scalar<DataType> c_k{};
    Scalar<DataType> c_theta{};
    Scalar<DataType> c_beta{};
    get(c_phi) = 4.0 * pow<2>(get(lapse)) / (denom_1 * get(conformal_factor));
    get(c_gamma) = pow<2>(get(lapse) * get(conformal_factor)) / denom_1;
    get(c_alpha) = -2.0 * get(lapse) / denom_2;
    get(c_k) = -4.0 * get(lapse) * sqrt(f) /
               (sqrt(3.0) * denom_2 * get(conformal_factor));
    get(c_theta) = 4.0 * sqrt(3.0 * f) * get(lapse) *
                   (-2.0 * f + get(lapse) * pow<2>(get(conformal_factor)) *
                                   (2.0 * get(lapse) - 1.0)) /
                   (get(conformal_factor) * denom_1 * denom_2);
    get(c_beta) = 2.0 / (sqrt(3.0 * f) * get(conformal_factor));

    ::tenex::evaluate(
        make_not_null(&u_scalar5_plus),
        unit_normal_one_form(ti::i) * auxiliary_field_b(ti::I) +
            c_phi() * unit_normal_vector(ti::I) * d_conformal_factor(ti::i) +
            c_k() * trace_extrinsic_curvature() + c_theta() * theta() +
            c_gamma() * unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
            c_alpha() * unit_normal_vector(ti::I) * d_lapse(ti::i) -
            c_beta() * unit_normal_vector(ti::I) * unit_normal_one_form(ti::j) *
                d_shift(ti::i, ti::J));

    ::tenex::evaluate(
        make_not_null(&u_scalar5_minus),
        unit_normal_one_form(ti::i) * auxiliary_field_b(ti::I) +
            c_phi() * unit_normal_vector(ti::I) * d_conformal_factor(ti::i) -
            c_k() * trace_extrinsic_curvature() - c_theta() * theta() +
            c_gamma() * unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
            c_alpha() * unit_normal_vector(ti::I) * d_lapse(ti::i) +
            c_beta() * unit_normal_vector(ti::I) * unit_normal_one_form(ti::j) *
                d_shift(ti::i, ti::J));
  } else {
    const auto n_plus =
        3 * shift_n * pow<2>(get(conformal_factor)) -
        get(conformal_factor) *
            sqrt(48.0 * f + 9 * pow<2>(shift_n * get(conformal_factor)));
    const auto n_minus =
        3 * shift_n * pow<2>(get(conformal_factor)) +
        get(conformal_factor) *
            sqrt(48.0 * f + 9 * pow<2>(shift_n * get(conformal_factor)));
    const auto d_1_plus = 8.0 * f + shift_n * n_minus -
                          6.0 * pow<2>(get(lapse) * get(conformal_factor));
    const auto d_1_minus = 8.0 * f + shift_n * n_plus -
                           6.0 * pow<2>(get(lapse) * get(conformal_factor));
    const auto d_2_plus = 8.0 * f + shift_n * n_minus -
                          12.0 * get(lapse) * pow<2>(get(conformal_factor));
    const auto d_2_minus = 8.0 * f + shift_n * n_plus -
                           12.0 * get(lapse) * pow<2>(get(conformal_factor));
    Scalar<DataType> c_phi_plus{};
    Scalar<DataType> c_phi_minus{};
    Scalar<DataType> c_k_plus{};
    Scalar<DataType> c_k_minus{};
    Scalar<DataType> c_theta_plus{};
    Scalar<DataType> c_theta_minus{};
    Scalar<DataType> c_gamma_plus{};
    Scalar<DataType> c_gamma_minus{};
    Scalar<DataType> c_alpha_plus{};
    Scalar<DataType> c_alpha_minus{};
    Scalar<DataType> c_beta_plus{};
    Scalar<DataType> c_beta_minus{};
    get(c_phi_plus) = -pow<2>(get(lapse)) * (shift_n * n_plus + 8.0 * f) /
                      (f * get(conformal_factor) * d_1_plus);
    get(c_phi_minus) = -pow<2>(get(lapse)) * (shift_n * n_minus + 8.0 * f) /
                       (f * get(conformal_factor) * d_1_minus);
    get(c_k_plus) = -4.0 * get(lapse) * n_plus /
                    (3.0 * pow<2>(get(conformal_factor)) * d_2_plus);
    get(c_k_minus) = -4.0 * get(lapse) * n_minus /
                     (3.0 * pow<2>(get(conformal_factor)) * d_2_minus);
    get(c_theta_plus) = 2.0 * get(lapse) * n_plus *
                        (8.0 * f + shift_n * n_minus +
                         4.0 * (1.0 - 2.0 * get(lapse)) * get(lapse) *
                             pow<2>(get(conformal_factor))) /
                        (pow<2>(get(conformal_factor)) * d_1_plus * d_2_plus);
    get(c_theta_minus) =
        2.0 * get(lapse) * n_minus *
        (8.0 * f + shift_n * n_plus +
         4.0 * (1.0 - 2.0 * get(lapse)) * get(lapse) *
             pow<2>(get(conformal_factor))) /
        (pow<2>(get(conformal_factor)) * d_1_minus * d_2_minus);
    get(c_gamma_plus) = -1.0 *
                        (shift_n * f * n_minus +
                         pow<2>(get(lapse)) * (shift_n * n_plus + 2.0 * f) *
                             pow<2>(get(conformal_factor))) /
                        (f * d_1_plus);
    get(c_gamma_minus) = -1.0 *
                         (shift_n * f * n_plus +
                          pow<2>(get(lapse)) * (shift_n * n_minus + 2.0 * f) *
                              pow<2>(get(conformal_factor))) /
                         (f * d_1_minus);
    get(c_alpha_plus) = get(lapse) * pow<2>(n_plus) /
                        (6.0 * f * pow<2>(get(conformal_factor)) * d_2_plus);
    get(c_alpha_minus) = get(lapse) * pow<2>(n_minus) /
                         (6.0 * f * pow<2>(get(conformal_factor)) * d_2_minus);
    get(c_beta_plus) = n_plus / (6.0 * f * pow<2>(get(conformal_factor)));
    get(c_beta_minus) = n_minus / (6.0 * f * pow<2>(get(conformal_factor)));
    ::tenex::evaluate(
        make_not_null(&u_scalar5_plus),
        unit_normal_one_form(ti::i) * auxiliary_field_b(ti::I) +
            c_phi_plus() * unit_normal_vector(ti::I) *
                d_conformal_factor(ti::i) +
            c_k_plus() * trace_extrinsic_curvature() +
            c_theta_plus() * theta() +
            c_gamma_plus() * unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
            c_alpha_plus() * unit_normal_vector(ti::I) * d_lapse(ti::i) +
            c_beta_plus() * unit_normal_vector(ti::I) *
                unit_normal_one_form(ti::j) * d_shift(ti::i, ti::J));
    ::tenex::evaluate(
        make_not_null(&u_scalar5_minus),
        unit_normal_one_form(ti::i) * auxiliary_field_b(ti::I) +
            c_phi_minus() * unit_normal_vector(ti::I) *
                d_conformal_factor(ti::i) +
            c_k_minus() * trace_extrinsic_curvature() +
            c_theta_minus() * theta() +
            c_gamma_minus() * unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
            c_alpha_minus() * unit_normal_vector(ti::I) * d_lapse(ti::i) +
            c_beta_minus() * unit_normal_vector(ti::I) *
                unit_normal_one_form(ti::j) * d_shift(ti::i, ti::J));
  }

  return expected_char_fields;
}
}  // namespace TestHelpers::Ccz4::fd::detail
