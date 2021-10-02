// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace GeneralizedHarmonic::Solutions {
template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpacetimeMetric<
    GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<gr::Tags::SpacetimeMetric<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);

  return {gr::spacetime_metric(lapse, shift, spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<
    GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
    Frame::Inertial>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<GeneralizedHarmonic::Tags::Phi<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrappedGr<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& deriv_shift =
      get<typename WrappedGr<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrappedGr<SolutionType>::DerivSpatialMetric>(
          intermediate_vars);

  return {GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Pi<
    GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
    Frame::Inertial>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<GeneralizedHarmonic::Tags::Pi<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& dt_lapse =
      get<typename WrappedGr<SolutionType>::TimeDerivLapse>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrappedGr<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& dt_shift =
      get<typename WrappedGr<SolutionType>::TimeDerivShift>(intermediate_vars);
  const auto& deriv_shift =
      get<typename WrappedGr<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& dt_spatial_metric =
      get<typename WrappedGr<SolutionType>::TimeDerivSpatialMetric>(
          intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrappedGr<SolutionType>::DerivSpatialMetric>(
          intermediate_vars);

  const auto phi =
      GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);

  return {GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi)};
}
}  // namespace GeneralizedHarmonic::Solutions

#define WRAPPED_GR_STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define WRAPPED_GR_INSTANTIATE(_, data)                                        \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpacetimeMetric<GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    WRAPPED_GR_STYPE(data)>::volume_dim,       \
                                Frame::Inertial, DataVector>>                  \
  GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(data)>::          \
      variables(                                                               \
          const tnsr::I<DataVector,                                            \
                        GeneralizedHarmonic::Solutions::WrappedGr<             \
                            WRAPPED_GR_STYPE(data)>::volume_dim>& /*x*/,       \
          tmpl::list<gr::Tags::SpacetimeMetric<                                \
              GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(      \
                  data)>::volume_dim,                                          \
              Frame::Inertial, DataVector>> /*meta*/,                          \
          const IntermediateVars& intermediate_vars) const;           \
  template tuples::TaggedTuple<                                                \
      GeneralizedHarmonic::Tags::Pi<GeneralizedHarmonic::Solutions::WrappedGr< \
                                        WRAPPED_GR_STYPE(data)>::volume_dim,   \
                                    Frame::Inertial>>                          \
  GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(data)>::          \
      variables(                                                               \
          const tnsr::I<DataVector,                                            \
                        GeneralizedHarmonic::Solutions::WrappedGr<             \
                            WRAPPED_GR_STYPE(data)>::volume_dim>& /*x*/,       \
          tmpl::list<GeneralizedHarmonic::Tags::Pi<                            \
              GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(      \
                  data)>::volume_dim,                                          \
              Frame::Inertial>> /*meta*/,                                      \
          const IntermediateVars& intermediate_vars) const;           \
  template tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<                 \
      GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(              \
          data)>::volume_dim,                                                  \
      Frame::Inertial>>                                                        \
  GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(data)>::          \
      variables(                                                               \
          const tnsr::I<DataVector,                                            \
                        GeneralizedHarmonic::Solutions::WrappedGr<             \
                            WRAPPED_GR_STYPE(data)>::volume_dim>& /*x*/,       \
          tmpl::list<GeneralizedHarmonic::Tags::Phi<                           \
              GeneralizedHarmonic::Solutions::WrappedGr<WRAPPED_GR_STYPE(      \
                  data)>::volume_dim,                                          \
              Frame::Inertial>> /*meta*/,                                      \
          const IntermediateVars& intermediate_vars) const;
