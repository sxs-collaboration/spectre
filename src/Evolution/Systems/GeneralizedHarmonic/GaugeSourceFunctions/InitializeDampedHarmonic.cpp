// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename Frame>
using deriv_tags = tmpl::list<
    ::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame>,
                  tmpl::size_t<Dim>, Frame>>;

template <size_t Dim, typename Frame>
using variables_tags =
    tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame>>;
}  // namespace

namespace GeneralizedHarmonic::gauges::Actions {
template <size_t Dim, bool UseRollon>
std::tuple<tnsr::a<DataVector, Dim, Frame::Inertial>,
           tnsr::ab<DataVector, Dim, Frame::Inertial>>
InitializeDampedHarmonic<Dim, UseRollon>::impl_rollon(
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian) noexcept {
  Variables<tmpl::list<
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>>
      temps{mesh.number_of_grid_points()};
  auto& spatial_metric =
      get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(temps);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
          temps);
  auto& shift = get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(temps);
  auto& lapse = get<gr::Tags::Lapse<DataVector>>(temps);
  auto& inverse_spacetime_metric =
      get<gr::Tags::InverseSpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
          temps);
  auto& da_spacetime_metric = get<
      gr::Tags::DerivativesOfSpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
      temps);

  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  determinant_and_inverse(
      make_not_null(&get<gr::Tags::DetSpatialMetric<DataVector>>(temps)),
      make_not_null(&inverse_spatial_metric), spatial_metric);
  gr::shift(make_not_null(&shift), spacetime_metric, inverse_spatial_metric);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);
  gr::inverse_spacetime_metric(make_not_null(&inverse_spacetime_metric), lapse,
                               shift, inverse_spatial_metric);
  GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      make_not_null(&da_spacetime_metric), lapse, shift, pi, phi);
  // H_a=-Gamma_a
  auto initial_gauge_h =
      trace_last_indices(gr::christoffel_first_kind(da_spacetime_metric),
                         inverse_spacetime_metric);
  for (size_t i = 0; i < initial_gauge_h.size(); ++i) {
    initial_gauge_h[i] *= -1.0;
  }

  // set time derivatives of InitialGaugeH = 0
  // NOTE: this will need to be generalized to handle numerical initial data
  // and analytic initial data whose gauge is not initially stationary.
  auto dt_initial_gauge_source =
      make_with_value<tnsr::a<DataVector, Dim, frame>>(initial_gauge_h, 0.);

  // compute spatial derivatives of InitialGaugeH
  // The `partial_derivatives` function does not support single Tensor input,
  // and so we must store the tensor in a Variables first.
  using InitialGaugeHVars = ::Variables<
      tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>>;
  InitialGaugeHVars initial_gauge_h_vars{mesh.number_of_grid_points()};

  get<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>(
      initial_gauge_h_vars) = initial_gauge_h;
  // compute spacetime derivatives of InitialGaugeH
  auto initial_d4_gauge_h =
      make_with_value<tnsr::ab<DataVector, Dim, Frame::Inertial>>(
          initial_gauge_h, 0.);
  GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<Dim, frame>::function(
      make_not_null(&initial_d4_gauge_h), std::move(dt_initial_gauge_source),
      get<::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                        tmpl::size_t<Dim>, frame>>(
          partial_derivatives<typename InitialGaugeHVars::tags_list>(
              initial_gauge_h_vars, mesh, inverse_jacobian)));

  return std::make_tuple(std::move(initial_gauge_h),
                         std::move(initial_d4_gauge_h));
}

template <size_t Dim, bool UseRollon>
void InitializeDampedHarmonic<Dim, UseRollon>::new_pi_from_gauge_h(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double sigma_r) noexcept {
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto det_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inverse_spatial_metric.first;
  const auto& inverse_spatial_metric = det_and_inverse_spatial_metric.second;
  const Scalar<DataVector> sqrt_det_spatial_metric{
      DataVector{sqrt(get(det_spatial_metric))}};
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(lapse);
  const auto spacetime_unit_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);

  const auto d_lapse =
      spatial_deriv_of_lapse(lapse, spacetime_unit_normal_vector, phi);
  const auto d_shift = spatial_deriv_of_shift(
      lapse, inverse_spacetime_metric, spacetime_unit_normal_vector, phi);
  const auto d_spatial_metric = deriv_spatial_metric(phi);

  const auto ex_curv =
      extrinsic_curvature(spacetime_unit_normal_vector, *pi, phi);
  const auto trace_ex_curv = trace(ex_curv, inverse_spatial_metric);
  const auto christoffel_first = gr::christoffel_first_kind(d_spatial_metric);
  const auto trace_christoffel_first =
      trace_last_indices(christoffel_first, inverse_spatial_metric);

  typename DampedHarmonicCompute<Frame::Inertial>::return_type
      gauge_h_and_deriv{get(lapse).size()};
  DampedHarmonicCompute<Frame::Inertial>::function(
      make_not_null(&gauge_h_and_deriv), lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, *pi, phi, coords, sigma_r);
  const auto& gauge_h =
      get<Tags::GaugeH<Dim, Frame::Inertial>>(gauge_h_and_deriv);

  // Compute lapse and shift time derivatives
  Scalar<DataVector> dt_lapse{};
  get(dt_lapse) =
      -get(lapse) * (get<0>(gauge_h) + get(lapse) * get(trace_ex_curv));
  for (size_t i = 0; i < Dim; ++i) {
    get(dt_lapse) +=
        shift.get(i) * (d_lapse.get(i) + get(lapse) * gauge_h.get(i + 1));
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> dt_shift{get(lapse).size(), 0.};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t k = 0; k < Dim; ++k) {
      dt_shift.get(i) += shift.get(k) * d_shift.get(k, i);
    }
    for (size_t j = 0; j < Dim; ++j) {
      dt_shift.get(i) +=
          get(lapse) * inverse_spatial_metric.get(i, j) *
          (get(lapse) * (gauge_h.get(j + 1) + trace_christoffel_first.get(j)) -
           d_lapse.get(j));
    }
  }

  GeneralizedHarmonic::pi(pi, lapse, dt_lapse, shift, dt_shift, spatial_metric,
                          time_deriv_of_spatial_metric(lapse, shift, phi, *pi),
                          phi);
}

template <size_t Dim, bool UseRollon>
template <typename Frame>
void InitializeDampedHarmonic<Dim, UseRollon>::
    DampedHarmonicRollonCompute<Frame>::function(
        const gsl::not_null<return_type*> h_and_d4_h,
        const db::item_type<Tags::InitialGaugeH<Dim, Frame>>& gauge_h_init,
        const db::item_type<Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>>&
            dgauge_h_init,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, Dim, Frame>& shift,
        const tnsr::a<DataVector, Dim, Frame>& spacetime_unit_normal_one_form,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
        const tnsr::aa<DataVector, Dim, Frame>& pi,
        const tnsr::iaa<DataVector, Dim, Frame>& phi, const double time,
        const double rollon_start_time, const double rollon_width,
        const tnsr::I<DataVector, Dim, Frame>& coords,
        const double sigma_r) noexcept {
  if (UNLIKELY(h_and_d4_h->number_of_grid_points() != get(lapse).size())) {
    h_and_d4_h->initialize(get(lapse).size());
  }
  damped_harmonic_rollon(
      make_not_null(&get<Tags::GaugeH<Dim, Frame>>(*h_and_d4_h)),
      make_not_null(&get<Tags::SpacetimeDerivGaugeH<Dim, Frame>>(*h_and_d4_h)),
      gauge_h_init, dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, time, coords, 1., 1.,
      1.,       // amp_coef_{L1, L2, S}
      4, 4, 4,  // exp_{L1, L2, S}
      rollon_start_time, rollon_width, sigma_r);
}

template <size_t Dim, bool UseRollon>
template <typename Frame>
void InitializeDampedHarmonic<Dim, UseRollon>::DampedHarmonicCompute<Frame>::
    function(
        const gsl::not_null<return_type*> h_and_d4_h,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, Dim, Frame>& shift,
        const tnsr::a<DataVector, Dim, Frame>& spacetime_unit_normal_one_form,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
        const tnsr::aa<DataVector, Dim, Frame>& pi,
        const tnsr::iaa<DataVector, Dim, Frame>& phi,
        const tnsr::I<DataVector, Dim, Frame>& coords,
        const double sigma_r) noexcept {
  if (UNLIKELY(h_and_d4_h->number_of_grid_points() != get(lapse).size())) {
    h_and_d4_h->initialize(get(lapse).size());
  }
  damped_harmonic(
      make_not_null(&get<Tags::GaugeH<Dim, Frame>>(*h_and_d4_h)),
      make_not_null(&get<Tags::SpacetimeDerivGaugeH<Dim, Frame>>(*h_and_d4_h)),
      lapse, shift, spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, coords, 1., 1.,
      1.,       // amp_coef_{L1, L2, S}
      4, 4, 4,  // exp_{L1, L2, S}
      sigma_r);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define USE_ROLLON(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class InitializeDampedHarmonic<DIM(data), USE_ROLLON(data)>::    \
      DampedHarmonicRollonCompute<Frame::Inertial>;                         \
  template class InitializeDampedHarmonic<                                  \
      DIM(data), USE_ROLLON(data)>::DampedHarmonicCompute<Frame::Inertial>; \
  template struct InitializeDampedHarmonic<DIM(data), USE_ROLLON(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef INSTANTIATE
#undef USE_ROLLON
#undef DIM
}  // namespace GeneralizedHarmonic::gauges::Actions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template Variables<deriv_tags<DIM(data), Frame::Inertial>>                 \
  partial_derivatives<variables_tags<DIM(data), Frame::Inertial>,            \
                      variables_tags<DIM(data), Frame::Inertial>, DIM(data), \
                      Frame::Inertial>(                                      \
      const Variables<variables_tags<DIM(data), Frame::Inertial>>& u,        \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,           \
                            Frame::Inertial>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
