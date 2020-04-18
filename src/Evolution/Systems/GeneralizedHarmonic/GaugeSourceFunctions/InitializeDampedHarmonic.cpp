// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/InitializeDampedHarmonic.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
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
template <size_t Dim>
std::tuple<tnsr::a<DataVector, Dim, Frame::Inertial>,
           tnsr::ab<DataVector, Dim, Frame::Inertial>>
InitializeDampedHarmonic<Dim>::impl(
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian) noexcept {
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  tnsr::abb<DataVector, Dim, Frame::Inertial> da_spacetime_metric{};
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
  tnsr::ab<DataVector, Dim, Frame::Inertial> initial_d4_gauge_h{};
  GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<Dim, frame>::function(
      make_not_null(&initial_d4_gauge_h), std::move(dt_initial_gauge_source),
      get<::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                        tmpl::size_t<Dim>, frame>>(
          partial_derivatives<typename InitialGaugeHVars::tags_list>(
              initial_gauge_h_vars, mesh, inverse_jacobian)));

  return std::make_tuple(std::move(initial_gauge_h),
                         std::move(initial_d4_gauge_h));
}

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
                            Frame::Inertial>& inverse_jacobian) noexcept;    \
  template struct GeneralizedHarmonic::gauges::Actions::                     \
      InitializeDampedHarmonic<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
