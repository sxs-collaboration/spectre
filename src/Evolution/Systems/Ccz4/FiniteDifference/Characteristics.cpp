// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"

#include "DataStructures/Tensor/ContractFirstNIndices.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Expressions/SquareRoot.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
namespace detail {
// Helper function to compute a rank-2 symmetric tensor from TT components
template <typename DataType, typename Frame>
tnsr::ii<DataType, Dim, Frame> reconstruct_symmetric_tensor_from_tt(
    const tnsr::ii<DataType, Dim, Frame>& tensor_tt,
    const tnsr::i<DataType, Dim, Frame>& tensor_perp_n,
    const Scalar<DataVector>& tensor_nn,
    const Scalar<DataVector>& tensor_transverse_trace,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& unit_normal_one_form) {
  const auto q_dd = projector_dd(spatial_metric, unit_normal_one_form);
  tnsr::ii<DataType, Dim, Frame> tensor{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&tensor),
      tensor_nn() * unit_normal_one_form(ti::i) * unit_normal_one_form(ti::j) +
          0.5 * tensor_transverse_trace() * q_dd(ti::i, ti::j) +
          unit_normal_one_form(ti::i) * tensor_perp_n(ti::j) +
          unit_normal_one_form(ti::j) * tensor_perp_n(ti::i) +
          tensor_tt(ti::i, ti::j));
  return tensor;
}

// Helper functions to compute the constants (Scalar)
// c_phi^\pm, c_gamma^\pm, c_alpha^\pm, c_k^\pm, c_theta^\pm, c_beta^\pm
template <typename DataType>
using coefficients_tags_list =
    tmpl::list<Tags::CPhi<DataType>, Tags::CGamma<DataType>,
               Tags::CAlpha<DataType>, Tags::CK<DataType>,
               Tags::CTheta<DataType>, Tags::CBeta<DataType>>;
template <typename DataType>
Variables<coefficients_tags_list<DataType>> compute_coefficients_plus(
    const Scalar<DataType>& lapse, const Scalar<DataType>& conformal_factor,
    const Scalar<DataType>& shift_dot_normal, const double f) {
  Variables<coefficients_tags_list<DataType>> coefficients{get(lapse).size()};

  if (::Ccz4::fd::System::shifting_shift) {
    const auto denom_1 =
        -4.0 * f + 3 * pow<2>(get(lapse) * get(conformal_factor));
    const auto denom_2 =
        -2.0 * f + 3 * get(lapse) * pow<2>(get(conformal_factor));
    auto& c_phi = get<Tags::CPhi<DataType>>(coefficients);
    auto& c_gamma = get<Tags::CGamma<DataType>>(coefficients);
    auto& c_alpha = get<Tags::CAlpha<DataType>>(coefficients);
    auto& c_k = get<Tags::CK<DataType>>(coefficients);
    auto& c_theta = get<Tags::CTheta<DataType>>(coefficients);
    auto& c_beta = get<Tags::CBeta<DataType>>(coefficients);
    get(c_phi) = 4.0 * pow<2>(get(lapse)) / denom_1 / get(conformal_factor);
    get(c_gamma) = pow<2>(get(lapse) * get(conformal_factor)) / denom_1;
    get(c_alpha) = -2.0 * get(lapse) / denom_2;
    get(c_k) = -4.0 * get(lapse) * sqrt(f) / denom_2 / get(conformal_factor) /
               sqrt(3.0);
    get(c_theta) = 4.0 * sqrt(3.0 * f) * get(lapse) *
                   (-2.0 * f + get(lapse) * (2 * get(lapse) - 1.0) *
                                   pow<2>(get(conformal_factor))) /
                   (denom_1 * denom_2 * get(conformal_factor));
    get(c_beta) = -2.0 / sqrt(3.0 * f) / get(conformal_factor);
    return coefficients;
  } else {
    const auto three_beta_n_phi_squared_plus_sqrt =
        get(conformal_factor) *
        (3.0 * get(shift_dot_normal) * get(conformal_factor) +
         sqrt(48.0 * f +
              9.0 * pow<2>(get(shift_dot_normal) * get(conformal_factor))));
    const auto three_beta_n_phi_squared_minus_sqrt =
        get(conformal_factor) *
        (3.0 * get(shift_dot_normal) * get(conformal_factor) -
         sqrt(48.0 * f +
              9.0 * pow<2>(get(shift_dot_normal) * get(conformal_factor))));
    const auto denom_1 =
        8.0 * f + get(shift_dot_normal) * three_beta_n_phi_squared_plus_sqrt -
        6 * pow<2>(get(lapse) * get(conformal_factor));
    const auto denom_2 =
        8.0 * f + get(shift_dot_normal) * three_beta_n_phi_squared_plus_sqrt -
        12.0 * get(lapse) * pow<2>(get(conformal_factor));
    auto& c_phi = get<Tags::CPhi<DataType>>(coefficients);
    auto& c_gamma = get<Tags::CGamma<DataType>>(coefficients);
    auto& c_alpha = get<Tags::CAlpha<DataType>>(coefficients);
    auto& c_k = get<Tags::CK<DataType>>(coefficients);
    auto& c_theta = get<Tags::CTheta<DataType>>(coefficients);
    auto& c_beta = get<Tags::CBeta<DataType>>(coefficients);
    get(c_phi) = -pow<2>(get(lapse)) *
                 (get(shift_dot_normal) * three_beta_n_phi_squared_minus_sqrt +
                  8.0 * f) /
                 (denom_1 * get(conformal_factor) * f);
    get(c_k) = -4.0 * get(lapse) * three_beta_n_phi_squared_minus_sqrt /
               (3.0 * denom_2 * pow<2>(get(conformal_factor)));
    get(c_theta) =
        2.0 * get(lapse) * three_beta_n_phi_squared_minus_sqrt *
        (8.0 * f + get(shift_dot_normal) * three_beta_n_phi_squared_plus_sqrt +
         4.0 * (1.0 - 2.0 * get(lapse)) * get(lapse) *
             pow<2>(get(conformal_factor))) /
        (denom_1 * denom_2 * pow<2>(get(conformal_factor)));
    get(c_gamma) =
        -(get(shift_dot_normal) * f * three_beta_n_phi_squared_plus_sqrt +
          pow<2>(get(lapse) * get(conformal_factor)) *
              (get(shift_dot_normal) * three_beta_n_phi_squared_minus_sqrt +
               2.0 * f)) /
        (denom_1 * f);
    get(c_alpha) = get(lapse) * pow<2>(three_beta_n_phi_squared_minus_sqrt) /
                   (6.0 * denom_2 * f * pow<2>(get(conformal_factor)));
    get(c_beta) = three_beta_n_phi_squared_minus_sqrt /
                  (6.0 * f * pow<2>(get(conformal_factor)));
    return coefficients;
  }
}

template <typename DataType>
using coefficients_tags_list =
    tmpl::list<Tags::CPhi<DataType>, Tags::CGamma<DataType>,
               Tags::CAlpha<DataType>, Tags::CK<DataType>,
               Tags::CTheta<DataType>, Tags::CBeta<DataType>>;
template <typename DataType>
Variables<coefficients_tags_list<DataType>> compute_coefficients_minus(
    const Scalar<DataType>& lapse, const Scalar<DataType>& conformal_factor,
    const Scalar<DataType>& shift_dot_normal, const double f) {
  Variables<coefficients_tags_list<DataType>> coefficients{get(lapse).size()};

  if (::Ccz4::fd::System::shifting_shift) {
    const auto denom_1 =
        -4.0 * f + 3 * pow<2>(get(lapse) * get(conformal_factor));
    const auto denom_2 =
        -2.0 * f + 3 * get(lapse) * pow<2>(get(conformal_factor));
    auto& c_phi = get<Tags::CPhi<DataType>>(coefficients);
    auto& c_gamma = get<Tags::CGamma<DataType>>(coefficients);
    auto& c_alpha = get<Tags::CAlpha<DataType>>(coefficients);
    auto& c_k = get<Tags::CK<DataType>>(coefficients);
    auto& c_theta = get<Tags::CTheta<DataType>>(coefficients);
    auto& c_beta = get<Tags::CBeta<DataType>>(coefficients);
    get(c_phi) = 4.0 * pow<2>(get(lapse)) / denom_1 / get(conformal_factor);
    get(c_gamma) = pow<2>(get(lapse) * get(conformal_factor)) / denom_1;
    get(c_alpha) = -2.0 * get(lapse) / denom_2;
    get(c_k) = 4.0 * get(lapse) * sqrt(f) / denom_2 / get(conformal_factor) /
               sqrt(3.0);
    get(c_theta) = -4.0 * sqrt(3.0 * f) * get(lapse) *
                   (-2.0 * f + get(lapse) * (2 * get(lapse) - 1.0) *
                                   pow<2>(get(conformal_factor))) /
                   (denom_1 * denom_2 * get(conformal_factor));
    get(c_beta) = 2.0 / sqrt(3.0 * f) / get(conformal_factor);
    return coefficients;
  } else {
    const auto three_beta_n_phi_squared_plus_sqrt =
        get(conformal_factor) *
        (3.0 * get(shift_dot_normal) * get(conformal_factor) +
         sqrt(48.0 * f +
              9.0 * pow<2>(get(shift_dot_normal) * get(conformal_factor))));
    const auto three_beta_n_phi_squared_minus_sqrt =
        get(conformal_factor) *
        (3.0 * get(shift_dot_normal) * get(conformal_factor) -
         sqrt(48.0 * f +
              9.0 * pow<2>(get(shift_dot_normal) * get(conformal_factor))));
    const auto denom_1 =
        get(shift_dot_normal) * three_beta_n_phi_squared_minus_sqrt + 8.0 * f -
        6 * pow<2>(get(lapse) * get(conformal_factor));
    const auto denom_2 =
        get(shift_dot_normal) * three_beta_n_phi_squared_minus_sqrt + 8.0 * f -
        12.0 * get(lapse) * pow<2>(get(conformal_factor));
    auto& c_phi = get<Tags::CPhi<DataType>>(coefficients);
    auto& c_gamma = get<Tags::CGamma<DataType>>(coefficients);
    auto& c_alpha = get<Tags::CAlpha<DataType>>(coefficients);
    auto& c_k = get<Tags::CK<DataType>>(coefficients);
    auto& c_theta = get<Tags::CTheta<DataType>>(coefficients);
    auto& c_beta = get<Tags::CBeta<DataType>>(coefficients);
    get(c_phi) = -(pow<2>(get(lapse)) *
                   (8.0 * f + get(shift_dot_normal) *
                                  three_beta_n_phi_squared_plus_sqrt)) /
                 (denom_1 * get(conformal_factor) * f);
    get(c_k) = -4.0 * get(lapse) * three_beta_n_phi_squared_plus_sqrt /
               (3.0 * denom_2 * pow<2>(get(conformal_factor)));
    get(c_theta) =
        2.0 * get(lapse) * three_beta_n_phi_squared_plus_sqrt *
        (8.0 * f + get(shift_dot_normal) * three_beta_n_phi_squared_minus_sqrt +
         4.0 * (1.0 - 2.0 * get(lapse)) * get(lapse) *
             pow<2>(get(conformal_factor))) /
        (denom_1 * denom_2 * pow<2>(get(conformal_factor)));
    get(c_gamma) =
        -(get(shift_dot_normal) * f * three_beta_n_phi_squared_minus_sqrt +
          pow<2>(get(lapse) * get(conformal_factor)) *
              (get(shift_dot_normal) * three_beta_n_phi_squared_plus_sqrt +
               2.0 * f)) /
        (denom_1 * f);
    get(c_alpha) = get(lapse) * pow<2>(three_beta_n_phi_squared_plus_sqrt) /
                   (6.0 * denom_2 * f * pow<2>(get(conformal_factor)));
    get(c_beta) = three_beta_n_phi_squared_plus_sqrt /
                  (6.0 * f * pow<2>(get(conformal_factor)));
    return coefficients;
  }
}
}  // namespace detail

// Helper function to compute the projector q_{ij} = gamma_{ij} - n_i n_j
template <typename DataType, typename Frame>
tnsr::ii<DataType, Dim, Frame> projector_dd(
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& unit_normal_one_form) {
  return gr::transverse_projection_operator(spatial_metric,
                                            unit_normal_one_form);
}

// Helper function to compute the TT part of a symmetric rank-2 tensor
template <typename DataType, typename Frame>
tnsr::ii<DataType, Dim, Frame> compute_tt_symmetric_tensor(
    const tnsr::ii<DataType, Dim, Frame>& tensor,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, Dim, Frame>& unit_normal_one_form) {
  const tnsr::ii<DataType, Dim, Frame> q_dd =
      projector_dd(spatial_metric, unit_normal_one_form);
  tnsr::iJ<DataType, Dim, Frame> q_dU{};
  ::tenex::evaluate<ti::i, ti::J>(
      make_not_null(&q_dU),
      q_dd(ti::i, ti::k) * inverse_spatial_metric(ti::K, ti::J));
  tnsr::II<DataType, Dim, Frame> q_UU{};
  ::tenex::evaluate<ti::I, ti::J>(
      make_not_null(&q_UU),
      inverse_spatial_metric(ti::I, ti::K) * q_dU(ti::k, ti::J));

  tnsr::ii<DataType, Dim, Frame> tensor_tt{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&tensor_tt),
      (q_dU(ti::i, ti::K) * q_dU(ti::j, ti::L) -
       0.5 * q_dd(ti::i, ti::j) * q_UU(ti::K, ti::L)) *
          tensor(ti::k, ti::l));
  return tensor_tt;
}

template <typename Frame>
std::array<DataVector, 16> characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& conformal_factor, const double f,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  auto char_speeds =
      make_with_value<typename Tags::CharacteristicSpeeds<DataVector>::type>(
          get(lapse), 0.);
  characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                        conformal_factor, f, unit_normal_one_form);
  return char_speeds;
}

template <typename Frame>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 16>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& conformal_factor, const double f,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form) {
  constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  const DataVector shift_dot_normal =
    get(dot_product(shift, unit_normal_one_form));
  // tensor sector
  (*char_speeds)[0] = -shift_dot_normal + get(lapse);  // tensor +
  (*char_speeds)[1] = -shift_dot_normal - get(lapse);  // tensor -
  // vector sector
  const auto sqrt_f_over_conformal_factor = sqrt(f) / get(conformal_factor);
  (*char_speeds)[2] =
      shifting_shift ? -shift_dot_normal : 0.0 * shift_dot_normal;
  (*char_speeds)[3] = (*char_speeds)[0];
  (*char_speeds)[4] = (*char_speeds)[1];
  if (shifting_shift) {
    (*char_speeds)[5] = -shift_dot_normal + sqrt_f_over_conformal_factor;
    (*char_speeds)[6] = -shift_dot_normal - sqrt_f_over_conformal_factor;
  } else {
    (*char_speeds)[5] =
        -0.5 * shift_dot_normal +
        sqrt(4.0 * f + pow<2>(shift_dot_normal * get(conformal_factor))) /
            (2.0 * get(conformal_factor));
    (*char_speeds)[6] =
        -0.5 * shift_dot_normal -
        sqrt(4.0 * f + pow<2>(shift_dot_normal * get(conformal_factor))) /
            (2.0 * get(conformal_factor));
  }
  // scalar sector
  (*char_speeds)[7] =
      shifting_shift ? -shift_dot_normal : 0.0 * shift_dot_normal;
  (*char_speeds)[8] = (*char_speeds)[0];
  (*char_speeds)[9] = (*char_speeds)[1];
  (*char_speeds)[10] = (*char_speeds)[0];
  (*char_speeds)[11] = (*char_speeds)[1];
  (*char_speeds)[12] = -shift_dot_normal + sqrt(2.0 * get(lapse));
  (*char_speeds)[13] = -shift_dot_normal - sqrt(2.0 * get(lapse));
  if (shifting_shift) {
    const auto v_s = (2.0) / sqrt(3.0) * sqrt_f_over_conformal_factor;
    (*char_speeds)[14] = -shift_dot_normal + v_s;
    (*char_speeds)[15] = -shift_dot_normal - v_s;
  } else {
    const auto v_s = sqrt(48.0 * f + 9.0 * pow<2>(shift_dot_normal *
                                                  get(conformal_factor))) /
                     (6.0 * get(conformal_factor));
    (*char_speeds)[14] = -0.5 * shift_dot_normal + v_s;
    (*char_speeds)[15] = -0.5 * shift_dot_normal - v_s;
  }
}

template <typename Frame>
typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type
characteristic_fields(
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, Dim, Frame>& a_tilde,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, Dim, Frame>& gamma_hat,
    const tnsr::I<DataVector, Dim, Frame>& auxiliary_field_b,
    const tnsr::ijj<DataVector, Dim, Frame>& d_conformal_spatial_metric,
    const tnsr::i<DataVector, Dim, Frame>& d_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame>& d_shift, const double f) {
  auto char_fields = make_with_value<
      typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type>(
      get(lapse), 0.);
  characteristic_fields(
      make_not_null(&char_fields), unit_normal_one_form,
      conformal_spatial_metric, conformal_factor, lapse, shift,
      trace_extrinsic_curvature, a_tilde, theta, gamma_hat, auxiliary_field_b,
      d_conformal_spatial_metric, d_conformal_factor, d_lapse, d_shift, f);
  return char_fields;
}

template <typename Frame>
void characteristic_fields(
    gsl::not_null<
        typename Tags::CharacteristicFields<DataVector, Dim, Frame>::type*>
        char_fields,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, Dim, Frame>& a_tilde,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, Dim, Frame>& gamma_hat,
    const tnsr::I<DataVector, Dim, Frame>& auxiliary_field_b,
    const tnsr::ijj<DataVector, Dim, Frame>& d_conformal_spatial_metric,
    const tnsr::i<DataVector, Dim, Frame>& d_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame>& d_shift, const double f) {
  const auto number_of_grid_points = get(lapse).size();
  if (UNLIKELY(number_of_grid_points != char_fields->number_of_grid_points())) {
    char_fields->initialize(number_of_grid_points);
  }
  constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  const Scalar<DataVector> shift_n = dot_product(shift, unit_normal_one_form);

  // Compute u_tnsr_plus and u_tnsr_minus
  tnsr::ii<DataVector, Dim, Frame> spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto a_tilde_tt = compute_tt_symmetric_tensor(
      a_tilde, spatial_metric, inverse_spatial_metric, unit_normal_one_form);
  const auto unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  tnsr::ii<DataVector, Dim, Frame> dn_conformal_spatial_metric{};
  contract_first_n_indices<1>(make_not_null(&dn_conformal_spatial_metric),
                              unit_normal_vector, d_conformal_spatial_metric);
  const auto dn_conformal_spatial_metric_tt =
      compute_tt_symmetric_tensor(dn_conformal_spatial_metric, spatial_metric,
                                  inverse_spatial_metric, unit_normal_one_form);
  auto& u_tnsr_plus =
      get<Tags::UTensorPlus<DataVector, Dim, Frame>>(*char_fields);
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&u_tnsr_plus),
      a_tilde_tt(ti::i, ti::j) +
          0.5 * dn_conformal_spatial_metric_tt(ti::i, ti::j));
  auto& u_tnsr_minus =
      get<Tags::UTensorMinus<DataVector, Dim, Frame>>(*char_fields);
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&u_tnsr_minus),
      a_tilde_tt(ti::i, ti::j) -
          0.5 * dn_conformal_spatial_metric_tt(ti::i, ti::j));

  // Compute u_vector1_zero
  const auto q_dd = projector_dd(spatial_metric, unit_normal_one_form);
  auto& u_vector1_zero =
      get<Tags::UVector1Zero<DataVector, Dim, Frame>>(*char_fields);
  ::tenex::evaluate<ti::i>(
      make_not_null(&u_vector1_zero),
      q_dd(ti::i, ti::j) * (auxiliary_field_b(ti::J) - gamma_hat(ti::J)));

  // Compute u_vector2_plus and u_vector2_minus
  // Potential optimization: rescale the characteristic fields so that
  // there is no division by conformal_factor
  tnsr::iJ<DataVector, Dim, Frame> q_dU{};
  ::tenex::evaluate<ti::i, ti::J>(
      make_not_null(&q_dU),
      q_dd(ti::i, ti::k) * inverse_spatial_metric(ti::K, ti::J));
  auto& u_vector2_plus =
      get<Tags::UVector2Plus<DataVector, Dim, Frame>>(*char_fields);
  ::tenex::evaluate<ti::i>(make_not_null(&u_vector2_plus),
                           q_dd(ti::i, ti::j) * gamma_hat(ti::J) -
                               unit_normal_vector(ti::J) * q_dU(ti::i, ti::K) /
                                   (conformal_factor() * conformal_factor() *
                                    conformal_factor() * conformal_factor()) *
                                   (dn_conformal_spatial_metric(ti::k, ti::j) +
                                    2.0 * a_tilde(ti::k, ti::j)));
  auto& u_vector2_minus =
      get<Tags::UVector2Minus<DataVector, Dim, Frame>>(*char_fields);
  ::tenex::evaluate<ti::i>(make_not_null(&u_vector2_minus),
                           q_dd(ti::i, ti::j) * gamma_hat(ti::J) -
                               unit_normal_vector(ti::J) * q_dU(ti::i, ti::K) /
                                   (conformal_factor() * conformal_factor() *
                                    conformal_factor() * conformal_factor()) *
                                   (dn_conformal_spatial_metric(ti::k, ti::j) -
                                    2.0 * a_tilde(ti::k, ti::j)));

  // Compute u_vector3_plus and u_vector3_minus
  auto& u_vector3_plus =
      get<Tags::UVector3Plus<DataVector, Dim, Frame>>(*char_fields);
  if (shifting_shift) {
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_plus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) -
            unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) /
                (sqrt(f) * conformal_factor()) * d_shift(ti::j, ti::K));
  } else {
    Scalar<DataVector> beta_n_times_phi_minus_sqrt_over_two_f{};
    get(beta_n_times_phi_minus_sqrt_over_two_f) =
        (get(shift_n) * get(conformal_factor) -
         sqrt(4.0 * f + pow<2>(get(shift_n) * get(conformal_factor)))) /
        (2.0 * f);
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_plus),
        q_dd(ti::i, ti::j) * gamma_hat(ti::J) * shift_n() *
                beta_n_times_phi_minus_sqrt_over_two_f() * conformal_factor() +
            q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) *
                d_shift(ti::j, ti::K) *
                beta_n_times_phi_minus_sqrt_over_two_f() / conformal_factor());
  }
  auto& u_vector3_minus =
      get<Tags::UVector3Minus<DataVector, Dim, Frame>>(*char_fields);
  if (shifting_shift) {
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_minus),
        q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) /
                (sqrt(f) * conformal_factor()) * d_shift(ti::j, ti::K));
  } else {
    Scalar<DataVector> beta_n_times_phi_plus_sqrt_over_two_f{};
    get(beta_n_times_phi_plus_sqrt_over_two_f) =
        (get(shift_n) * get(conformal_factor) +
         sqrt(4.0 * f + pow<2>(get(shift_n) * get(conformal_factor)))) /
        (2.0 * f);
    ::tenex::evaluate<ti::i>(
        make_not_null(&u_vector3_minus),
        q_dd(ti::i, ti::j) * gamma_hat(ti::J) * shift_n() *
                beta_n_times_phi_plus_sqrt_over_two_f() * conformal_factor() +
            q_dd(ti::i, ti::j) * auxiliary_field_b(ti::J) +
            unit_normal_vector(ti::J) * q_dd(ti::i, ti::k) *
                d_shift(ti::j, ti::K) *
                beta_n_times_phi_plus_sqrt_over_two_f() / conformal_factor());
  }

  // Compute u_scalar1_zero
  auto& u_scalar1_zero = get<Tags::UScalar1Zero<DataVector>>(*char_fields);
  ::tenex::evaluate(make_not_null(&u_scalar1_zero),
                    unit_normal_one_form(ti::i) *
                        (auxiliary_field_b(ti::I) - gamma_hat(ti::I)));

  // Compute u_scalar2_plus and u_scalar2_minus
  auto& u_scalar2_plus = get<Tags::UScalar2Plus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar2_plus),
      unit_normal_vector(ti::I) * unit_normal_vector(ti::J) *
              (a_tilde(ti::i, ti::j) +
               0.5 * dn_conformal_spatial_metric(ti::i, ti::j)) +
          2.0 * conformal_factor() *
              (unit_normal_vector(ti::I) * d_conformal_factor(ti::i) -
               conformal_factor() * trace_extrinsic_curvature() / 3.0));
  auto& u_scalar2_minus = get<Tags::UScalar2Minus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar2_minus),
      unit_normal_vector(ti::I) * unit_normal_vector(ti::J) *
              (a_tilde(ti::i, ti::j) -
               0.5 * dn_conformal_spatial_metric(ti::i, ti::j)) -
          2.0 * conformal_factor() *
              (unit_normal_vector(ti::I) * d_conformal_factor(ti::i) +
               conformal_factor() * trace_extrinsic_curvature() / 3.0));

  // Compute u_scalar3_plus and u_scalar3_minus
  auto& u_scalar3_plus = get<Tags::UScalar3Plus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar3_plus),
      unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
          2.0 / (conformal_factor() * conformal_factor()) *
              (2.0 / conformal_factor() * d_conformal_factor(ti::i) *
                   unit_normal_vector(ti::I) -
               theta()));
  auto& u_scalar3_minus = get<Tags::UScalar3Minus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar3_minus),
      unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
          2.0 / (conformal_factor() * conformal_factor()) *
              (2.0 / conformal_factor() * d_conformal_factor(ti::i) *
                   unit_normal_vector(ti::I) +
               theta()));

  // Compute u_scalar4_plus and u_scalar4_minus
  auto& u_scalar4_plus = get<Tags::UScalar4Plus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar4_plus),
      unit_normal_vector(ti::I) * d_lapse(ti::i) +
          sqrt(2.0 * lapse()) * (trace_extrinsic_curvature() - 2.0 * theta()));
  auto& u_scalar4_minus = get<Tags::UScalar4Minus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar4_minus),
      unit_normal_vector(ti::I) * d_lapse(ti::i) -
          sqrt(2.0 * lapse()) * (trace_extrinsic_curvature() - 2.0 * theta()));

  // Compute u_scalar5_plus and u_scalar5_minus
  const auto coefficients_plus =
      detail::compute_coefficients_plus(lapse, conformal_factor, shift_n, f);
  const auto& c_phi_plus = get<Tags::CPhi<DataVector>>(coefficients_plus);
  const auto& c_gamma_plus = get<Tags::CGamma<DataVector>>(coefficients_plus);
  const auto& c_alpha_plus = get<Tags::CAlpha<DataVector>>(coefficients_plus);
  const auto& c_k_plus = get<Tags::CK<DataVector>>(coefficients_plus);
  const auto& c_theta_plus = get<Tags::CTheta<DataVector>>(coefficients_plus);
  const auto& c_beta_plus = get<Tags::CBeta<DataVector>>(coefficients_plus);
  auto& u_scalar5_plus = get<Tags::UScalar5Plus<DataVector>>(*char_fields);
  ::tenex::evaluate(
      make_not_null(&u_scalar5_plus),
      unit_normal_one_form(ti::i) * auxiliary_field_b(ti::I) +
          c_phi_plus() * unit_normal_vector(ti::I) * d_conformal_factor(ti::i) +
          c_k_plus() * trace_extrinsic_curvature() + c_theta_plus() * theta() +
          c_gamma_plus() * unit_normal_one_form(ti::i) * gamma_hat(ti::I) +
          c_alpha_plus() * unit_normal_vector(ti::I) * d_lapse(ti::i) +
          c_beta_plus() * unit_normal_vector(ti::I) *
              unit_normal_one_form(ti::j) * d_shift(ti::i, ti::J));

  const auto coefficients_minus =
      detail::compute_coefficients_minus(lapse, conformal_factor, shift_n, f);
  const auto& c_phi_minus = get<Tags::CPhi<DataVector>>(coefficients_minus);
  const auto& c_gamma_minus = get<Tags::CGamma<DataVector>>(coefficients_minus);
  const auto& c_alpha_minus = get<Tags::CAlpha<DataVector>>(coefficients_minus);
  const auto& c_k_minus = get<Tags::CK<DataVector>>(coefficients_minus);
  const auto& c_theta_minus = get<Tags::CTheta<DataVector>>(coefficients_minus);
  const auto& c_beta_minus = get<Tags::CBeta<DataVector>>(coefficients_minus);
  auto& u_scalar5_minus = get<Tags::UScalar5Minus<DataVector>>(*char_fields);
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

template <typename Frame>
typename Tags::EvolvedSpaceFromCharacteristicFields<DataVector, Dim,
                                                    Frame>::type
evolved_space_from_characteristic_fields(
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_plus,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector1_zero,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_minus,
    const Scalar<DataVector>& u_scalar1_zero,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const Scalar<DataVector>& u_scalar4_plus,
    const Scalar<DataVector>& u_scalar4_minus,
    const Scalar<DataVector>& u_scalar5_plus,
    const Scalar<DataVector>& u_scalar5_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift, const double f) {
  auto evolved_space =
      make_with_value<typename Tags::EvolvedSpaceFromCharacteristicFields<
          DataVector, Dim, Frame>::type>(get(u_scalar1_zero), 0.);
  evolved_space_from_characteristic_fields(
      make_not_null(&evolved_space), u_tnsr_plus, u_tnsr_minus, u_vector1_zero,
      u_vector2_plus, u_vector2_minus, u_vector3_plus, u_vector3_minus,
      u_scalar1_zero, u_scalar2_plus, u_scalar2_minus, u_scalar3_plus,
      u_scalar3_minus, u_scalar4_plus, u_scalar4_minus, u_scalar5_plus,
      u_scalar5_minus, unit_normal_one_form, conformal_spatial_metric,
      conformal_factor, lapse, shift, f);
  return evolved_space;
}

template <typename Frame>
void evolved_space_from_characteristic_fields(
    gsl::not_null<typename Tags::EvolvedSpaceFromCharacteristicFields<
        DataVector, Dim, Frame>::type*>
        evolved_space,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_plus,
    const tnsr::ii<DataVector, Dim, Frame>& u_tnsr_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector1_zero,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector2_minus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_plus,
    const tnsr::i<DataVector, Dim, Frame>& u_vector3_minus,
    const Scalar<DataVector>& u_scalar1_zero,
    const Scalar<DataVector>& u_scalar2_plus,
    const Scalar<DataVector>& u_scalar2_minus,
    const Scalar<DataVector>& u_scalar3_plus,
    const Scalar<DataVector>& u_scalar3_minus,
    const Scalar<DataVector>& u_scalar4_plus,
    const Scalar<DataVector>& u_scalar4_minus,
    const Scalar<DataVector>& u_scalar5_plus,
    const Scalar<DataVector>& u_scalar5_minus,
    const tnsr::i<DataVector, Dim, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, Dim, Frame>& conformal_spatial_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame>& shift, const double f) {
  const auto number_of_grid_points = get(u_scalar1_zero).size();
  if (UNLIKELY(number_of_grid_points !=
               evolved_space->number_of_grid_points())) {
    evolved_space->initialize(number_of_grid_points);
  }
  constexpr bool shifting_shift = ::Ccz4::fd::System::shifting_shift;
  const Scalar<DataVector> shift_n = dot_product(shift, unit_normal_one_form);

  // Reconstruct gamma_hat
  const auto coefficients_plus =
      detail::compute_coefficients_plus(lapse, conformal_factor, shift_n, f);
  const auto& c_phi_plus = get<Tags::CPhi<DataVector>>(coefficients_plus);
  const auto& c_gamma_plus = get<Tags::CGamma<DataVector>>(coefficients_plus);
  const auto& c_alpha_plus = get<Tags::CAlpha<DataVector>>(coefficients_plus);
  const auto& c_k_plus = get<Tags::CK<DataVector>>(coefficients_plus);
  const auto& c_theta_plus = get<Tags::CTheta<DataVector>>(coefficients_plus);
  const auto& c_beta_plus = get<Tags::CBeta<DataVector>>(coefficients_plus);
  const auto coefficients_minus =
      detail::compute_coefficients_minus(lapse, conformal_factor, shift_n, f);
  const auto& c_phi_minus = get<Tags::CPhi<DataVector>>(coefficients_minus);
  const auto& c_gamma_minus = get<Tags::CGamma<DataVector>>(coefficients_minus);
  const auto& c_alpha_minus = get<Tags::CAlpha<DataVector>>(coefficients_minus);
  const auto& c_k_minus = get<Tags::CK<DataVector>>(coefficients_minus);
  const auto& c_theta_minus = get<Tags::CTheta<DataVector>>(coefficients_minus);
  const auto& c_beta_minus = get<Tags::CBeta<DataVector>>(coefficients_minus);
  Scalar<DataVector> gamma_hat_n{};
  Scalar<DataVector> source_plus{};
  get(source_plus) =
      get(u_scalar5_plus) - get(u_scalar1_zero) -
      pow<3>(get(conformal_factor)) / 8.0 * get(c_phi_plus) *
          (get(u_scalar3_plus) + get(u_scalar3_minus)) -
      0.5 * get(c_alpha_plus) * (get(u_scalar4_plus) + get(u_scalar4_minus)) -
      get(c_k_plus) * ((get(u_scalar4_plus) - get(u_scalar4_minus)) /
                           (2.0 * sqrt(2.0 * get(lapse))) -
                       0.5 * pow<2>(get(conformal_factor)) *
                           (get(u_scalar3_plus) - get(u_scalar3_minus))) +
      0.25 * get(c_theta_plus) * pow<2>(get(conformal_factor)) *
          (get(u_scalar3_plus) - get(u_scalar3_minus));
  Scalar<DataVector> source_minus{};
  get(source_minus) =
      get(u_scalar5_minus) - get(u_scalar1_zero) -
      pow<3>(get(conformal_factor)) / 8.0 * get(c_phi_minus) *
          (get(u_scalar3_plus) + get(u_scalar3_minus)) -
      0.5 * get(c_alpha_minus) * (get(u_scalar4_plus) + get(u_scalar4_minus)) -
      get(c_k_minus) * ((get(u_scalar4_plus) - get(u_scalar4_minus)) /
                            (2.0 * sqrt(2.0 * get(lapse))) -
                        0.5 * pow<2>(get(conformal_factor)) *
                            (get(u_scalar3_plus) - get(u_scalar3_minus))) +
      0.25 * get(c_theta_minus) * pow<2>(get(conformal_factor)) *
          (get(u_scalar3_plus) - get(u_scalar3_minus));
  Scalar<DataVector> denom{};
  get(denom) = get(c_beta_plus) *
                   ((1 + get(c_gamma_minus)) -
                    get(c_phi_minus) * pow<3>(get(conformal_factor)) / 4.0) -
               get(c_beta_minus) *
                   ((1 + get(c_gamma_plus)) -
                    get(c_phi_plus) * pow<3>(get(conformal_factor)) / 4.0);
  get(gamma_hat_n) = (get(c_beta_plus) * get(source_minus) -
                      get(c_beta_minus) * get(source_plus)) /
                     get(denom);

  Scalar<DataVector> c_gamma_vec_plus{};
  Scalar<DataVector> c_gamma_vec_minus{};
  Scalar<DataVector> c_beta_vec_plus{};
  Scalar<DataVector> c_beta_vec_minus{};
  if (shifting_shift) {
    get(c_gamma_vec_plus) = 0.0 * get(conformal_factor);
    get(c_gamma_vec_minus) = 0.0 * get(conformal_factor);
    get(c_beta_vec_plus) = -1.0 / get(conformal_factor) / sqrt(f);
    get(c_beta_vec_minus) = 1.0 / get(conformal_factor) / sqrt(f);
  } else {
    get(c_beta_vec_plus) =
        (get(shift_n) -
         sqrt(4.0 * f + pow<2>(get(shift_n) * get(conformal_factor))) /
             get(conformal_factor)) /
        (2.0 * f);
    get(c_beta_vec_minus) =
        (get(shift_n) +
         sqrt(4.0 * f + pow<2>(get(shift_n) * get(conformal_factor))) /
             get(conformal_factor)) /
        (2.0 * f);
    get(c_gamma_vec_plus) =
        get(shift_n) * pow<2>(get(conformal_factor)) * get(c_beta_vec_plus);
    get(c_gamma_vec_minus) =
        get(shift_n) * pow<2>(get(conformal_factor)) * get(c_beta_vec_minus);
  }

  Scalar<DataVector> denom_vec{};
  get(denom_vec) = get(c_beta_vec_plus) * (1.0 + get(c_gamma_vec_minus)) -
                   get(c_beta_vec_minus) * (1.0 + get(c_gamma_vec_plus));
  tnsr::i<DataVector, Dim, Frame> gamma_hat_perp{};
  ::tenex::evaluate<ti::j>(
      make_not_null(&gamma_hat_perp),
      (c_beta_vec_plus() * u_vector3_minus(ti::j) -
       c_beta_vec_minus() * u_vector3_plus(ti::j) +
       (c_beta_vec_minus() - c_beta_vec_plus()) * u_vector1_zero(ti::j)) /
          denom_vec());

  tnsr::ii<DataVector, Dim, Frame> spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  auto& gamma_hat =
      get<::Ccz4::Tags::GammaHat<DataVector, Dim, Frame>>(*evolved_space);
  ::tenex::evaluate<ti::I>(make_not_null(&gamma_hat),
                           inverse_spatial_metric(ti::I, ti::J) *
                               // gamma_hat_perp part
                               (gamma_hat_perp(ti::j) +
                                // gamma_hat_n part
                                gamma_hat_n() * unit_normal_one_form(ti::j)));

  // Reconstruct dn_conformal_spatial_metric
  tnsr::ii<DataVector, Dim, Frame> dn_conformal_spatial_metric_tt{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&dn_conformal_spatial_metric_tt),
      u_tnsr_plus(ti::i, ti::j) - u_tnsr_minus(ti::i, ti::j));
  tnsr::i<DataVector, Dim, Frame> dn_conformal_spatial_metric_perp_n{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&dn_conformal_spatial_metric_perp_n),
      conformal_factor() * conformal_factor() * conformal_factor() *
          conformal_factor() *
          (gamma_hat_perp(ti::i) -
           0.5 * (u_vector2_plus(ti::i) + u_vector2_minus(ti::i))));
  Scalar<DataVector> dn_conformal_spatial_metric_nn{};
  get(dn_conformal_spatial_metric_nn) =
      (get(u_scalar2_plus) - get(u_scalar2_minus)) -
      pow<4>(get(conformal_factor)) *
          (0.5 * (get(u_scalar3_plus) + get(u_scalar3_minus)) -
           get(gamma_hat_n));
  // By the Jacobi formula,
  // dn_conformal_spatial_metric_transverse_trace =
  // -dn_conformal_spatial_metric_nn
  Scalar<DataVector> dn_conformal_spatial_metric_transverse_trace{};
  get(dn_conformal_spatial_metric_transverse_trace) =
      -get(dn_conformal_spatial_metric_nn);
  get<Tags::DnConformalMetric<DataVector, Dim, Frame>>(*evolved_space) =
      detail::reconstruct_symmetric_tensor_from_tt(
          dn_conformal_spatial_metric_tt, dn_conformal_spatial_metric_perp_n,
          dn_conformal_spatial_metric_nn,
          dn_conformal_spatial_metric_transverse_trace, spatial_metric,
          unit_normal_one_form);

  // Reconstruct a_tilde
  tnsr::ii<DataVector, Dim, Frame> a_tilde_tt{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&a_tilde_tt),
      0.5 * (u_tnsr_plus(ti::i, ti::j) + u_tnsr_minus(ti::i, ti::j)));
  tnsr::i<DataVector, Dim, Frame> a_tilde_perp_n{};
  ::tenex::evaluate<ti::i>(
      make_not_null(&a_tilde_perp_n),
      -0.25 * conformal_factor() * conformal_factor() * conformal_factor() *
          conformal_factor() *
          (u_vector2_plus(ti::i) - u_vector2_minus(ti::i)));
  Scalar<DataVector> a_tilde_nn{};
  get(a_tilde_nn) = 0.5 * (get(u_scalar2_plus) + get(u_scalar2_minus)) +
                    2.0 / 3.0 * pow<2>(get(conformal_factor)) *
                        ((get(u_scalar4_plus) - get(u_scalar4_minus)) /
                             (2.0 * sqrt(2.0 * get(lapse))) -
                         0.5 * pow<2>(get(conformal_factor)) *
                             (get(u_scalar3_plus) - get(u_scalar3_minus)));
  // Since a_tilde is trace-free, a_tilde_transverse_trace = - a_tilde_nn
  Scalar<DataVector> a_tilde_transverse_trace{};
  get(a_tilde_transverse_trace) = -get(a_tilde_nn);
  get<::Ccz4::Tags::ATilde<DataVector, Dim, Frame>>(*evolved_space) =
      detail::reconstruct_symmetric_tensor_from_tt(
          a_tilde_tt, a_tilde_perp_n, a_tilde_nn, a_tilde_transverse_trace,
          spatial_metric, unit_normal_one_form);

  // Reconstruct auxiliary_field_b
  auto& auxiliary_field_b =
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim, Frame>>(
          *evolved_space);
  ::tenex::evaluate<ti::I>(
      make_not_null(&auxiliary_field_b),
      inverse_spatial_metric(ti::I, ti::J) *
          // b_perp part
          (u_vector1_zero(ti::j) + gamma_hat_perp(ti::j) +
           // b_n part
           (u_scalar1_zero() + gamma_hat_n()) * unit_normal_one_form(ti::j)));

  // Reconstruct dn_shift
  Scalar<DataVector> dn_shift_n{};
  get(dn_shift_n) =
      (get(source_plus) *
           (1.0 + get(c_gamma_minus) -
            get(c_phi_minus) * pow<3>(get(conformal_factor)) / 4.0) -
       get(source_minus) *
           (1.0 + get(c_gamma_plus) -
            get(c_phi_plus) * pow<3>(get(conformal_factor)) / 4.0)) /
      get(denom);
  auto& dn_shift = get<Tags::DnShift<DataVector, Dim, Frame>>(*evolved_space);
  ::tenex::evaluate<ti::I>(
      make_not_null(&dn_shift),
      inverse_spatial_metric(ti::I, ti::J) *
          // dn_shift_perp part
          (((1.0 + c_gamma_vec_minus()) * u_vector3_plus(ti::j) -
            (1.0 + c_gamma_vec_plus()) * u_vector3_minus(ti::j) +
            (c_gamma_vec_plus() - c_gamma_vec_minus()) *
                u_vector1_zero(ti::j)) /
               denom_vec() +
           // dn_shift_n part
           dn_shift_n() * unit_normal_one_form(ti::j)));

  // Reconstruct theta
  auto& theta = get<::Ccz4::Tags::Theta<DataVector>>(*evolved_space);
  get(theta) = -0.25 * pow<2>(get(conformal_factor)) *
               (get(u_scalar3_plus) - get(u_scalar3_minus));

  // Reconstruct dn_lapse
  auto& dn_lapse = get<Tags::DnLapse<DataVector>>(*evolved_space);
  get(dn_lapse) = 0.5 * (get(u_scalar4_plus) + get(u_scalar4_minus));

  // Reconstruct trace_extrinsic_curvature
  auto& trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(*evolved_space);
  get(trace_extrinsic_curvature) =
      2.0 * get(theta) + (get(u_scalar4_plus) - get(u_scalar4_minus)) /
                             (2.0 * sqrt(2.0 * get(lapse)));

  // Reconstruct dn_conformal_factor
  auto& dn_conformal_factor =
      get<Tags::DnConformalFactor<DataVector>>(*evolved_space);
  get(dn_conformal_factor) =
      0.25 * pow<3>(get(conformal_factor)) *
      (0.5 * (get(u_scalar3_plus) + get(u_scalar3_minus)) - get(gamma_hat_n));
}
}  // namespace Ccz4::fd

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>                    \
  Ccz4::fd::projector_dd<DataVector, FRAME(data)>(                             \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& spatial_metric,  \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form);                                               \
  template tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>                    \
  Ccz4::fd::compute_tt_symmetric_tensor<DataVector, FRAME(data)>(              \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& tensor,          \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& spatial_metric,  \
      const tnsr::II<DataVector, Ccz4::fd::Dim, FRAME(data)>&                  \
          inverse_spatial_metric,                                              \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form);                                               \
  template std::array<DataVector, 16>                                          \
  Ccz4::fd::characteristic_speeds<FRAME(data)>(                                \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,            \
      const Scalar<DataVector>& conformal_factor, const double f,              \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form);                                               \
  template void Ccz4::fd::characteristic_speeds<FRAME(data)>(                  \
      const gsl::not_null<std::array<DataVector, 16>*> char_speeds,            \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,            \
      const Scalar<DataVector>& conformal_factor, const double f,              \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form);                                               \
  template struct Ccz4::fd::CharacteristicSpeedsCompute<FRAME(data)>;          \
  template                                                                     \
      typename Ccz4::fd::Tags::CharacteristicFields<DataVector, Ccz4::fd::Dim, \
                                                    FRAME(data)>::type         \
      Ccz4::fd::characteristic_fields<FRAME(data)>(                            \
          const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&               \
              unit_normal_one_form,                                            \
          const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>&              \
              conformal_spatial_metric,                                        \
          const Scalar<DataVector>& conformal_factor,                          \
          const Scalar<DataVector>& lapse,                                     \
          const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,        \
          const Scalar<DataVector>& trace_extrinsic_curvature,                 \
          const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& a_tilde,     \
          const Scalar<DataVector>& theta,                                     \
          const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& gamma_hat,    \
          const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>&               \
              auxiliary_field_b,                                               \
          const tnsr::ijj<DataVector, Ccz4::fd::Dim, FRAME(data)>&             \
              d_conformal_spatial_metric,                                      \
          const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&               \
              d_conformal_factor,                                              \
          const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& d_lapse,      \
          const tnsr::iJ<DataVector, Ccz4::fd::Dim, FRAME(data)>& d_shift,     \
          const double f);                                                     \
  template void Ccz4::fd::characteristic_fields<FRAME(data)>(                  \
      gsl::not_null<typename Ccz4::fd::Tags::CharacteristicFields<             \
          DataVector, Ccz4::fd::Dim, FRAME(data)>::type*>                      \
          char_fields,                                                         \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form,                                                \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>&                  \
          conformal_spatial_metric,                                            \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,            \
      const Scalar<DataVector>& trace_extrinsic_curvature,                     \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& a_tilde,         \
      const Scalar<DataVector>& theta,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& gamma_hat,        \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          auxiliary_field_b,                                                   \
      const tnsr::ijj<DataVector, Ccz4::fd::Dim, FRAME(data)>&                 \
          d_conformal_spatial_metric,                                          \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          d_conformal_factor,                                                  \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& d_lapse,          \
      const tnsr::iJ<DataVector, Ccz4::fd::Dim, FRAME(data)>& d_shift,         \
      const double f);                                                         \
  template struct Ccz4::fd::CharacteristicFieldsCompute<FRAME(data)>;          \
  template typename Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<      \
      DataVector, Ccz4::fd::Dim, FRAME(data)>::type                            \
  Ccz4::fd::evolved_space_from_characteristic_fields<FRAME(data)>(             \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_tnsr_plus,     \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_tnsr_minus,    \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector1_zero,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector2_plus,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector2_minus,  \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector3_plus,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector3_minus,  \
      const Scalar<DataVector>& u_scalar1_zero,                                \
      const Scalar<DataVector>& u_scalar2_plus,                                \
      const Scalar<DataVector>& u_scalar2_minus,                               \
      const Scalar<DataVector>& u_scalar3_plus,                                \
      const Scalar<DataVector>& u_scalar3_minus,                               \
      const Scalar<DataVector>& u_scalar4_plus,                                \
      const Scalar<DataVector>& u_scalar4_minus,                               \
      const Scalar<DataVector>& u_scalar5_plus,                                \
      const Scalar<DataVector>& u_scalar5_minus,                               \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form,                                                \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>&                  \
          conformal_spatial_metric,                                            \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,            \
      const double f);                                                         \
  template void                                                                \
  Ccz4::fd::evolved_space_from_characteristic_fields<FRAME(data)>(             \
      gsl::not_null<                                                           \
          typename Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<       \
              DataVector, Ccz4::fd::Dim, FRAME(data)>::type*>                  \
          evolved_space,                                                       \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_tnsr_plus,     \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_tnsr_minus,    \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector1_zero,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector2_plus,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector2_minus,  \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector3_plus,   \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>& u_vector3_minus,  \
      const Scalar<DataVector>& u_scalar1_zero,                                \
      const Scalar<DataVector>& u_scalar2_plus,                                \
      const Scalar<DataVector>& u_scalar2_minus,                               \
      const Scalar<DataVector>& u_scalar3_plus,                                \
      const Scalar<DataVector>& u_scalar3_minus,                               \
      const Scalar<DataVector>& u_scalar4_plus,                                \
      const Scalar<DataVector>& u_scalar4_minus,                               \
      const Scalar<DataVector>& u_scalar5_plus,                                \
      const Scalar<DataVector>& u_scalar5_minus,                               \
      const tnsr::i<DataVector, Ccz4::fd::Dim, FRAME(data)>&                   \
          unit_normal_one_form,                                                \
      const tnsr::ii<DataVector, Ccz4::fd::Dim, FRAME(data)>&                  \
          conformal_spatial_metric,                                            \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, Ccz4::fd::Dim, FRAME(data)>& shift,            \
      const double f);                                                         \
  template struct Ccz4::fd::EvolvedSpaceFromCharacteristicFieldsCompute<FRAME( \
      data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef FRAME
