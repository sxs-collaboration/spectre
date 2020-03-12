// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename TagsList>
class Variables;

// IWYU pragma: no_forward_declare DataVector
// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
namespace GeneralizedHarmonic_detail {
template <typename FieldTag>
db::const_item_type<FieldTag> weight_char_field(
    const db::const_item_type<FieldTag>& char_field_int,
    const DataVector& char_speed_int,
    const db::const_item_type<FieldTag>& char_field_ext,
    const DataVector& char_speed_ext) noexcept {
  const DataVector& char_speed_avg{0.5 * (char_speed_int + char_speed_ext)};
  db::const_item_type<FieldTag> weighted_char_field = char_field_int;
  auto weighted_char_field_it = weighted_char_field.begin();
  for (auto int_it = char_field_int.begin(), ext_it = char_field_ext.begin();
       int_it != char_field_int.end();
       ++int_it, ++ext_it, ++weighted_char_field_it) {
    *weighted_char_field_it *= char_speed_avg * step_function(char_speed_avg);
    *weighted_char_field_it +=
        char_speed_avg * step_function(-char_speed_avg) * *ext_it;
  }

  return weighted_char_field;
}

// Function that performs the upwind weighting. Inputs are the char fields
// and speeds in the interior and exterior. At each point, each returned
// field is the product of the interior char field and its char speed
// (if the char speed is outgoing) or the product of the exterior char field
// and its char speed (if the char speed is incoming).
template <size_t Dim>
db::const_item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>
weight_char_fields(
    const db::const_item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>&
        char_fields_int,
    const db::const_item_type<Tags::CharacteristicSpeeds<Dim, Frame::Inertial>>&
        char_speeds_int,
    const db::const_item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>&
        char_fields_ext,
    const db::const_item_type<Tags::CharacteristicSpeeds<Dim, Frame::Inertial>>&
        char_speeds_ext) noexcept {
  const auto& u_psi_int =
      get<Tags::UPsi<Dim, Frame::Inertial>>(char_fields_int);
  const auto& u_zero_int =
      get<Tags::UZero<Dim, Frame::Inertial>>(char_fields_int);
  const auto& u_plus_int =
      get<Tags::UPlus<Dim, Frame::Inertial>>(char_fields_int);
  const auto& u_minus_int =
      get<Tags::UMinus<Dim, Frame::Inertial>>(char_fields_int);

  const DataVector& char_speed_u_psi_int{char_speeds_int[0]};
  const DataVector& char_speed_u_zero_int{char_speeds_int[1]};
  const DataVector& char_speed_u_plus_int{char_speeds_int[2]};
  const DataVector& char_speed_u_minus_int{char_speeds_int[3]};

  const auto& u_psi_ext =
      get<Tags::UPsi<Dim, Frame::Inertial>>(char_fields_ext);
  const auto& u_zero_ext =
      get<Tags::UZero<Dim, Frame::Inertial>>(char_fields_ext);
  const auto& u_plus_ext =
      get<Tags::UPlus<Dim, Frame::Inertial>>(char_fields_ext);
  const auto& u_minus_ext =
      get<Tags::UMinus<Dim, Frame::Inertial>>(char_fields_ext);

  const DataVector& char_speed_u_psi_ext{char_speeds_ext[0]};
  const DataVector& char_speed_u_zero_ext{char_speeds_ext[1]};
  const DataVector& char_speed_u_plus_ext{char_speeds_ext[2]};
  const DataVector& char_speed_u_minus_ext{char_speeds_ext[3]};

  auto weighted_char_fields = make_with_value<
      db::const_item_type<Tags::CharacteristicFields<Dim, Frame::Inertial>>>(
      char_speed_u_psi_int, 0.0);

  get<Tags::UPsi<Dim, Frame::Inertial>>(weighted_char_fields) =
      weight_char_field<GeneralizedHarmonic::Tags::UPsi<Dim, Frame::Inertial>>(
          u_psi_int, char_speed_u_psi_int, u_psi_ext, char_speed_u_psi_ext);
  get<Tags::UZero<Dim, Frame::Inertial>>(weighted_char_fields) =
      weight_char_field<GeneralizedHarmonic::Tags::UZero<Dim, Frame::Inertial>>(
          u_zero_int, char_speed_u_zero_int, u_zero_ext, char_speed_u_zero_ext);
  get<Tags::UPlus<Dim, Frame::Inertial>>(weighted_char_fields) =
      weight_char_field<GeneralizedHarmonic::Tags::UPlus<Dim, Frame::Inertial>>(
          u_plus_int, char_speed_u_plus_int, u_plus_ext, char_speed_u_plus_ext);
  get<Tags::UMinus<Dim, Frame::Inertial>>(weighted_char_fields) =
      weight_char_field<
          GeneralizedHarmonic::Tags::UMinus<Dim, Frame::Inertial>>(
          u_minus_int, char_speed_u_minus_int, u_minus_ext,
          char_speed_u_minus_ext);

  return weighted_char_fields;
}
}  // namespace GeneralizedHarmonic_detail

/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> dt_phi,
    const tnsr::aa<DataVector, Dim>& spacetime_metric,
    const tnsr::aa<DataVector, Dim>& pi, const tnsr::iaa<DataVector, Dim>& phi,
    const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim>& d_pi,
    const tnsr::ijaa<DataVector, Dim>& d_phi, const Scalar<DataVector>& gamma0,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::a<DataVector, Dim>& gauge_function,
    const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::AA<DataVector, Dim>& inverse_spacetime_metric,
    const tnsr::a<DataVector, Dim>& trace_christoffel,
    const tnsr::abb<DataVector, Dim>& christoffel_first_kind,
    const tnsr::Abb<DataVector, Dim>& christoffel_second_kind,
    const tnsr::A<DataVector, Dim>& normal_spacetime_vector,
    const tnsr::a<DataVector, Dim>& normal_spacetime_one_form) {
  const size_t n_pts = shift.begin()->size();

  const DataVector gamma12 = gamma1.get() * gamma2.get();

  tnsr::Iaa<DataVector, Dim> phi_1_up{DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t n = 0; n < Dim; ++n) {
        for (size_t nu = mu; nu < Dim + 1; ++nu) {
          phi_1_up.get(m, mu, nu) +=
              inverse_spatial_metric.get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  tnsr::abC<DataVector, Dim> phi_3_up{DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        for (size_t beta = 0; beta < Dim + 1; ++beta) {
          phi_3_up.get(m, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  tnsr::aB<DataVector, Dim> pi_2_up{DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      for (size_t beta = 0; beta < Dim + 1; ++beta) {
        pi_2_up.get(nu, alpha) +=
            inverse_spacetime_metric.get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  tnsr::abC<DataVector, Dim> christoffel_first_kind_3_up{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        for (size_t beta = 0; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up.get(mu, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) *
              christoffel_first_kind.get(mu, nu, beta);
        }
      }
    }
  }

  tnsr::a<DataVector, Dim> pi_dot_normal_spacetime_vector{
      DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      pi_dot_normal_spacetime_vector.get(mu) +=
          normal_spacetime_vector.get(nu) * pi.get(nu, mu);
    }
  }

  DataVector pi_contract_two_normal_spacetime_vectors{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_contract_two_normal_spacetime_vectors +=
        normal_spacetime_vector.get(mu) *
        pi_dot_normal_spacetime_vector.get(mu);
  }

  tnsr::ia<DataVector, Dim> phi_dot_normal_spacetime_vector{
      DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t mu = 0; mu < Dim + 1; ++mu) {
        phi_dot_normal_spacetime_vector.get(n, nu) +=
            normal_spacetime_vector.get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim> phi_contract_two_normal_spacetime_vectors{
      DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      phi_contract_two_normal_spacetime_vectors.get(n) +=
          normal_spacetime_vector.get(mu) *
          phi_dot_normal_spacetime_vector.get(n, mu);
    }
  }

  tnsr::iaa<DataVector, Dim> three_index_constraint{DataVector(n_pts, 0.)};
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        three_index_constraint.get(n, mu, nu) =
            d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim> one_index_constraint{DataVector(n_pts, 0.)};
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    one_index_constraint.get(nu) =
        gauge_function.get(nu) + trace_christoffel.get(nu);
  }

  DataVector normal_dot_one_index_constraint{DataVector(n_pts, 0.)};
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    normal_dot_one_index_constraint +=
        normal_spacetime_vector.get(mu) * one_index_constraint.get(mu);
  }

  const DataVector gamma1p1 = 1.0 + gamma1.get();

  tnsr::aa<DataVector, Dim> shift_dot_three_index_constraint{
      DataVector(n_pts, 0.)};
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        shift_dot_three_index_constraint.get(mu, nu) +=
            shift.get(m) * three_index_constraint.get(m, mu, nu);
      }
    }
  }

  // Here are the actual equations

  // Equation for dt_spacetime_metric
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_spacetime_metric->get(mu, nu) = -lapse.get() * pi.get(mu, nu);
      dt_spacetime_metric->get(mu, nu) +=
          gamma1p1 * shift_dot_three_index_constraint.get(mu, nu);
      for (size_t m = 0; m < Dim; ++m) {
        dt_spacetime_metric->get(mu, nu) += shift.get(m) * phi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_pi
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      dt_pi->get(mu, nu) =
          -spacetime_deriv_gauge_function.get(mu, nu) -
          spacetime_deriv_gauge_function.get(nu, mu) -
          0.5 * pi_contract_two_normal_spacetime_vectors * pi.get(mu, nu) +
          gamma0.get() * (normal_spacetime_one_form.get(mu) *
                              one_index_constraint.get(nu) +
                          normal_spacetime_one_form.get(nu) *
                              one_index_constraint.get(mu)) -
          gamma0.get() * spacetime_metric.get(mu, nu) *
              normal_dot_one_index_constraint;

      for (size_t delta = 0; delta < Dim + 1; ++delta) {
        dt_pi->get(mu, nu) += 2 * christoffel_second_kind.get(delta, mu, nu) *
                                  gauge_function.get(delta) -
                              2 * pi.get(mu, delta) * pi_2_up.get(nu, delta);

        for (size_t n = 0; n < Dim; ++n) {
          dt_pi->get(mu, nu) +=
              2 * phi_1_up.get(n, mu, delta) * phi_3_up.get(n, nu, delta);
        }

        for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
          dt_pi->get(mu, nu) -=
              2. * christoffel_first_kind_3_up.get(mu, alpha, delta) *
              christoffel_first_kind_3_up.get(nu, delta, alpha);
        }
      }

      for (size_t m = 0; m < Dim; ++m) {
        dt_pi->get(mu, nu) -=
            pi_dot_normal_spacetime_vector.get(m + 1) * phi_1_up.get(m, mu, nu);

        for (size_t n = 0; n < Dim; ++n) {
          dt_pi->get(mu, nu) -=
              inverse_spatial_metric.get(m, n) * d_phi.get(m, n, mu, nu);
        }
      }

      dt_pi->get(mu, nu) *= lapse.get();

      dt_pi->get(mu, nu) +=
          gamma12 * shift_dot_three_index_constraint.get(mu, nu);

      for (size_t m = 0; m < Dim; ++m) {
        // DualFrame term
        dt_pi->get(mu, nu) += shift.get(m) * d_pi.get(m, mu, nu);
      }
    }
  }

  // Equation for dt_phi
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        dt_phi->get(i, mu, nu) =
            0.5 * pi.get(mu, nu) *
                phi_contract_two_normal_spacetime_vectors.get(i) -
            d_pi.get(i, mu, nu) +
            gamma2.get() * three_index_constraint.get(i, mu, nu);
        for (size_t n = 0; n < Dim; ++n) {
          dt_phi->get(i, mu, nu) +=
              phi_dot_normal_spacetime_vector.get(i, n + 1) *
              phi_1_up.get(n, mu, nu);
        }

        dt_phi->get(i, mu, nu) *= lapse.get();
        for (size_t m = 0; m < Dim; ++m) {
          dt_phi->get(i, mu, nu) += shift.get(m) * d_phi.get(m, i, mu, nu);
        }
      }
    }
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<tnsr::aa<DataVector, Dim>*>
        spacetime_metric_normal_dot_flux,
    const gsl::not_null<tnsr::aa<DataVector, Dim>*> pi_normal_dot_flux,
    const gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi_normal_dot_flux,
    const tnsr::aa<DataVector, Dim>& spacetime_metric,
    const tnsr::aa<DataVector, Dim>& pi, const tnsr::iaa<DataVector, Dim>& phi,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
    const tnsr::i<DataVector, Dim>& unit_normal) noexcept {
  const auto shift_dot_normal = get(dot_product(shift, unit_normal));

  auto normal_dot_phi = make_with_value<tnsr::aa<DataVector, Dim>>(gamma1, 0.);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      for (size_t i = 0; i < Dim; ++i) {
        for (size_t j = 0; j < Dim; ++j) {
          normal_dot_phi.get(mu, nu) += inverse_spatial_metric.get(i, j) *
                                        unit_normal.get(j) * phi.get(i, mu, nu);
        }
      }
    }
  }

  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      spacetime_metric_normal_dot_flux->get(mu, nu) =
          -(1. + get(gamma1)) * spacetime_metric.get(mu, nu) * shift_dot_normal;
    }
  }

  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      pi_normal_dot_flux->get(mu, nu) =
          -shift_dot_normal *
              (get(gamma1) * get(gamma2) * spacetime_metric.get(mu, nu) +
               pi.get(mu, nu)) +
          get(lapse) * normal_dot_phi.get(mu, nu);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_normal_dot_flux->get(i, mu, nu) =
            get(lapse) * (unit_normal.get(i) * pi.get(mu, nu) -
                          get(gamma2) * unit_normal.get(i) *
                              spacetime_metric.get(mu, nu)) -
            shift_dot_normal * phi.get(i, mu, nu);
      }
    }
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
      *packaged_data) = spacetime_metric;
  get<Tags::Pi<Dim, Frame::Inertial>>(*packaged_data) = pi;
  get<Tags::Phi<Dim, Frame::Inertial>>(*packaged_data) = phi;
  get<gr::Tags::Lapse<DataVector>>(*packaged_data) = lapse;
  get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(*packaged_data) =
      shift;
  get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
      *packaged_data) = inverse_spatial_metric;
  get<Tags::ConstraintGamma1>(*packaged_data) = gamma1;
  get<Tags::ConstraintGamma2>(*packaged_data) = gamma2;
  get<::Tags::Normalized<
      domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>(
      *packaged_data) = interface_unit_normal;
}

template <size_t Dim>
void UpwindFlux<Dim>::operator()(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        psi_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        pi_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        phi_normal_dot_numerical_flux,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_int,
    const Scalar<DataVector>& lapse_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_int,
    const Scalar<DataVector>& gamma1_int, const Scalar<DataVector>& gamma2_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_ext,
    const Scalar<DataVector>& lapse_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_ext,
    const Scalar<DataVector>& gamma1_ext, const Scalar<DataVector>& gamma2_ext,
    const tnsr::i<DataVector, Dim,
                  Frame::Inertial>& /*interface_unit_normal_ext*/) const
    noexcept {
  const Scalar<DataVector> gamma1_avg{0.5 *
                                      (get(gamma1_int) + get(gamma1_ext))};
  const Scalar<DataVector> gamma2_avg{0.5 *
                                      (get(gamma2_int) + get(gamma2_ext))};

  const auto char_fields_int = characteristic_fields(
      gamma2_avg, inverse_spatial_metric_int, spacetime_metric_int, pi_int,
      phi_int, interface_unit_normal_int);
  const auto char_speeds_int = characteristic_speeds(
      gamma1_avg, lapse_int, shift_int, interface_unit_normal_int);
  const auto char_fields_ext = characteristic_fields(
      gamma2_avg, inverse_spatial_metric_ext, spacetime_metric_ext, pi_ext,
      phi_ext, interface_unit_normal_int);
  const auto char_speeds_ext = characteristic_speeds(
      gamma1_avg, lapse_ext, shift_ext, interface_unit_normal_int);

  const auto weighted_char_fields =
      GeneralizedHarmonic_detail::weight_char_fields<Dim>(
          char_fields_int, char_speeds_int, char_fields_ext, char_speeds_ext);

  const auto weighted_evolved_fields =
      evolved_fields_from_characteristic_fields(
          gamma2_avg,
          get<Tags::UPsi<Dim, Frame::Inertial>>(weighted_char_fields),
          get<Tags::UZero<Dim, Frame::Inertial>>(weighted_char_fields),
          get<Tags::UPlus<Dim, Frame::Inertial>>(weighted_char_fields),
          get<Tags::UMinus<Dim, Frame::Inertial>>(weighted_char_fields),
          interface_unit_normal_int);

  *psi_normal_dot_numerical_flux =
      get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>>(
          weighted_evolved_fields);
  *pi_normal_dot_numerical_flux =
      get<Tags::Pi<Dim, Frame::Inertial>>(weighted_evolved_fields);
  *phi_normal_dot_numerical_flux =
      get<Tags::Phi<Dim, Frame::Inertial>>(weighted_evolved_fields);
}
/// \endcond
}  // namespace GeneralizedHarmonic

// Explicit instantiations of structs defined in `Equations.cpp` as well as of
// `partial_derivatives` function for use in the computation of spatial
// derivatives of `gradients_tags`, and of the initial gauge source function
// (needed in `Initialize.hpp`).
/// \cond
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

using derivative_frame = Frame::Inertial;

template <size_t Dim>
using derivative_tags_initial_gauge =
    tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, derivative_frame>>;

template <size_t Dim>
using variables_tags_initial_gauge =
    tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, derivative_frame>>;

template <size_t Dim>
using derivative_tags =
    typename GeneralizedHarmonic::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename GeneralizedHarmonic::System<Dim>::variables_tag::tags_list;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                 \
  template struct GeneralizedHarmonic::ComputeDuDt<DIM(data)>;               \
  template struct GeneralizedHarmonic::ComputeNormalDotFluxes<DIM(data)>;    \
  template struct GeneralizedHarmonic::UpwindFlux<DIM(data)>;                \
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;   \
  template Variables<db::wrap_tags_in<                                       \
      ::Tags::deriv, derivative_tags_initial_gauge<DIM(data)>,               \
      tmpl::size_t<DIM(data)>, derivative_frame>>                            \
  partial_derivatives<derivative_tags_initial_gauge<DIM(data)>,              \
                      variables_tags_initial_gauge<DIM(data)>, DIM(data),    \
                      derivative_frame>(                                     \
      const Variables<variables_tags_initial_gauge<DIM(data)>>& u,           \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
/// \endcond
