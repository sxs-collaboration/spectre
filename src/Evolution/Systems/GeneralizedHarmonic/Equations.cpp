// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename TagsList>
class Variables;

// IWYU pragma: no_forward_declare DataVector
// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
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
    const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function) {
  const size_t n_pts = spacetime_metric[0].size();

  Variables<tmpl::list<
      Tags::Gamma1Gamma2, Tags::PiTwoNormals, Tags::NormalDotOneIndexConstraint,
      Tags::Gamma1Plus1, Tags::PiOneNormal<Dim>,
      Tags::GaugeConstraint<Dim, Frame::Inertial>, Tags::PhiTwoNormals<Dim>,
      Tags::ShiftDotThreeIndexConstraint<Dim>, Tags::PhiOneNormal<Dim>,
      Tags::PiSecondIndexUp<Dim>,
      Tags::ThreeIndexConstraint<Dim, Frame::Inertial>,
      Tags::PhiFirstIndexUp<Dim>, Tags::PhiThirdIndexUp<Dim>,
      Tags::SpacetimeChristoffelFirstKindThirdIndexUp<Dim>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame::Inertial,
                                               DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                   DataVector>,
      gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>>
      buffer(n_pts);

  auto& lapse = get<gr::Tags::Lapse<DataVector>>(buffer);
  auto& shift = get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(buffer);
  auto& spatial_metric =
      get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(buffer);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
          buffer);
  auto& det_spatial_metric =
      get<gr::Tags::DetSpatialMetric<DataVector>>(buffer);
  auto& inverse_spacetime_metric =
      get<gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          buffer);
  auto& christoffel_first_kind =
      get<gr::Tags::SpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                  DataVector>>(buffer);
  auto& christoffel_second_kind =
      get<gr::Tags::SpacetimeChristoffelSecondKind<Dim, Frame::Inertial,
                                                   DataVector>>(buffer);
  auto& trace_christoffel =
      get<gr::Tags::TraceSpacetimeChristoffelFirstKind<Dim, Frame::Inertial,
                                                       DataVector>>(buffer);
  auto& normal_spacetime_vector =
      get<gr::Tags::SpacetimeNormalVector<Dim, Frame::Inertial, DataVector>>(
          buffer);
  auto& normal_spacetime_one_form =
      get<gr::Tags::SpacetimeNormalOneForm<Dim, Frame::Inertial, DataVector>>(
          buffer);
  auto& da_spacetime_metric = get<
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
      buffer);

  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inverse_spatial_metric),
                          spatial_metric);
  gr::shift(make_not_null(&shift), spacetime_metric, inverse_spatial_metric);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);
  gr::inverse_spacetime_metric(make_not_null(&inverse_spacetime_metric), lapse,
                               shift, inverse_spatial_metric);
  GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      make_not_null(&da_spacetime_metric), lapse, shift, pi, phi);
  gr::christoffel_first_kind(make_not_null(&christoffel_first_kind),
                             da_spacetime_metric);
  raise_or_lower_first_index(make_not_null(&christoffel_second_kind),
                             christoffel_first_kind, inverse_spacetime_metric);
  trace_last_indices(make_not_null(&trace_christoffel), christoffel_first_kind,
                     inverse_spacetime_metric);
  gr::spacetime_normal_vector(make_not_null(&normal_spacetime_vector), lapse,
                              shift);
  gr::spacetime_normal_one_form(make_not_null(&normal_spacetime_one_form),
                                lapse);

  get(get<Tags::Gamma1Gamma2>(buffer)) = gamma1.get() * gamma2.get();
  const DataVector& gamma12 = get(get<Tags::Gamma1Gamma2>(buffer));

  tnsr::Iaa<DataVector, Dim>& phi_1_up =
      get<Tags::PhiFirstIndexUp<Dim>>(buffer);
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        phi_1_up.get(m, mu, nu) =
            inverse_spatial_metric.get(m, 0) * phi.get(0, mu, nu);
        for (size_t n = 1; n < Dim; ++n) {
          phi_1_up.get(m, mu, nu) +=
              inverse_spatial_metric.get(m, n) * phi.get(n, mu, nu);
        }
      }
    }
  }

  tnsr::iaB<DataVector, Dim>& phi_3_up =
      get<Tags::PhiThirdIndexUp<Dim>>(buffer);
  for (size_t m = 0; m < Dim; ++m) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        phi_3_up.get(m, nu, alpha) =
            inverse_spacetime_metric.get(alpha, 0) * phi.get(m, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          phi_3_up.get(m, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) * phi.get(m, nu, beta);
        }
      }
    }
  }

  tnsr::aB<DataVector, Dim>& pi_2_up = get<Tags::PiSecondIndexUp<Dim>>(buffer);
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
      pi_2_up.get(nu, alpha) =
          inverse_spacetime_metric.get(alpha, 0) * pi.get(nu, 0);
      for (size_t beta = 1; beta < Dim + 1; ++beta) {
        pi_2_up.get(nu, alpha) +=
            inverse_spacetime_metric.get(alpha, beta) * pi.get(nu, beta);
      }
    }
  }

  tnsr::abC<DataVector, Dim>& christoffel_first_kind_3_up =
      get<Tags::SpacetimeChristoffelFirstKindThirdIndexUp<Dim>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      for (size_t alpha = 0; alpha < Dim + 1; ++alpha) {
        christoffel_first_kind_3_up.get(mu, nu, alpha) =
            inverse_spacetime_metric.get(alpha, 0) *
            christoffel_first_kind.get(mu, nu, 0);
        for (size_t beta = 1; beta < Dim + 1; ++beta) {
          christoffel_first_kind_3_up.get(mu, nu, alpha) +=
              inverse_spacetime_metric.get(alpha, beta) *
              christoffel_first_kind.get(mu, nu, beta);
        }
      }
    }
  }

  tnsr::a<DataVector, Dim>& pi_dot_normal_spacetime_vector =
      get<Tags::PiOneNormal<Dim>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    pi_dot_normal_spacetime_vector.get(mu) =
        get<0>(normal_spacetime_vector) * pi.get(0, mu);
    for (size_t nu = 1; nu < Dim + 1; ++nu) {
      pi_dot_normal_spacetime_vector.get(mu) +=
          normal_spacetime_vector.get(nu) * pi.get(nu, mu);
    }
  }

  DataVector& pi_contract_two_normal_spacetime_vectors =
      get(get<Tags::PiTwoNormals>(buffer));
  pi_contract_two_normal_spacetime_vectors =
      get<0>(normal_spacetime_vector) * get<0>(pi_dot_normal_spacetime_vector);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    pi_contract_two_normal_spacetime_vectors +=
        normal_spacetime_vector.get(mu) *
        pi_dot_normal_spacetime_vector.get(mu);
  }

  tnsr::ia<DataVector, Dim>& phi_dot_normal_spacetime_vector =
      get<Tags::PhiOneNormal<Dim>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t nu = 0; nu < Dim + 1; ++nu) {
      phi_dot_normal_spacetime_vector.get(n, nu) =
          get<0>(normal_spacetime_vector) * phi.get(n, 0, nu);
      for (size_t mu = 1; mu < Dim + 1; ++mu) {
        phi_dot_normal_spacetime_vector.get(n, nu) +=
            normal_spacetime_vector.get(mu) * phi.get(n, mu, nu);
      }
    }
  }

  tnsr::i<DataVector, Dim>& phi_contract_two_normal_spacetime_vectors =
      get<Tags::PhiTwoNormals<Dim>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    phi_contract_two_normal_spacetime_vectors.get(n) =
        get<0>(normal_spacetime_vector) *
        phi_dot_normal_spacetime_vector.get(n, 0);
    for (size_t mu = 1; mu < Dim + 1; ++mu) {
      phi_contract_two_normal_spacetime_vectors.get(n) +=
          normal_spacetime_vector.get(mu) *
          phi_dot_normal_spacetime_vector.get(n, mu);
    }
  }

  tnsr::iaa<DataVector, Dim>& three_index_constraint =
      get<Tags::ThreeIndexConstraint<Dim, Frame::Inertial>>(buffer);
  for (size_t n = 0; n < Dim; ++n) {
    for (size_t mu = 0; mu < Dim + 1; ++mu) {
      for (size_t nu = mu; nu < Dim + 1; ++nu) {
        three_index_constraint.get(n, mu, nu) =
            d_spacetime_metric.get(n, mu, nu) - phi.get(n, mu, nu);
      }
    }
  }

  tnsr::a<DataVector, Dim>& one_index_constraint =
      get<Tags::GaugeConstraint<Dim, Frame::Inertial>>(buffer);
  for (size_t nu = 0; nu < Dim + 1; ++nu) {
    one_index_constraint.get(nu) =
        gauge_function.get(nu) + trace_christoffel.get(nu);
  }

  DataVector& normal_dot_one_index_constraint =
      get(get<Tags::NormalDotOneIndexConstraint>(buffer));
  normal_dot_one_index_constraint =
      get<0>(normal_spacetime_vector) * get<0>(one_index_constraint);
  for (size_t mu = 1; mu < Dim + 1; ++mu) {
    normal_dot_one_index_constraint +=
        normal_spacetime_vector.get(mu) * one_index_constraint.get(mu);
  }

  get(get<Tags::Gamma1Plus1>(buffer)) = 1.0 + gamma1.get();
  const DataVector& gamma1p1 = get(get<Tags::Gamma1Plus1>(buffer));

  tnsr::aa<DataVector, Dim>& shift_dot_three_index_constraint =
      get<Tags::ShiftDotThreeIndexConstraint<Dim>>(buffer);
  for (size_t mu = 0; mu < Dim + 1; ++mu) {
    for (size_t nu = mu; nu < Dim + 1; ++nu) {
      shift_dot_three_index_constraint.get(mu, nu) =
          get<0>(shift) * three_index_constraint.get(0, mu, nu);
      for (size_t m = 1; m < Dim; ++m) {
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
    const tnsr::aa<DataVector, Dim>& spacetime_metric) noexcept {
  destructive_resize_components(pi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(phi_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  destructive_resize_components(spacetime_metric_normal_dot_flux,
                                get<0, 0>(spacetime_metric).size());
  for (size_t storage_index = 0; storage_index < pi_normal_dot_flux->size();
       ++storage_index) {
    (*pi_normal_dot_flux)[storage_index] = 0.0;
    (*spacetime_metric_normal_dot_flux)[storage_index] = 0.0;
  }

  for (size_t storage_index = 0; storage_index < phi_normal_dot_flux->size();
       ++storage_index) {
    (*phi_normal_dot_flux)[storage_index] = 0.0;
  }
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
