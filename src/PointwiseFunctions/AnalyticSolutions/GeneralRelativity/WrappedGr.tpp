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

namespace gh::Solutions {
template <typename SolutionType>
WrappedGr<SolutionType>::WrappedGr(CkMigrateMessage* msg)
    : InitialData(msg), SolutionType(msg) {}

template <typename SolutionType>
std::unique_ptr<evolution::initial_data::InitialData>
WrappedGr<SolutionType>::get_clone() const {
  return std::make_unique<WrappedGr<SolutionType>>(*this);
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpacetimeMetric<
    DataVector, gh::Solutions::WrappedGr<SolutionType>::volume_dim>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector,
                  gh::Solutions::WrappedGr<SolutionType>::volume_dim>& /*x*/,
    tmpl::list<gr::Tags::SpacetimeMetric<
        DataVector,
        gh::Solutions::WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector,
                          gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
          intermediate_vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);

  return {gr::spacetime_metric(lapse, shift, spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<gh::Tags::Phi<DataVector,
    gh::Solutions::WrappedGr<SolutionType>::volume_dim>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, gh::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<gh::Tags::Phi<DataVector,
        gh::Solutions::WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrappedGr<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift =
      get<gr::Tags::Shift<DataVector,
                          gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
          intermediate_vars);
  const auto& deriv_shift =
      get<typename WrappedGr<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrappedGr<SolutionType>::DerivSpatialMetric>(
          intermediate_vars);

  return {gh::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<gh::Tags::Pi<DataVector,
    gh::Solutions::WrappedGr<SolutionType>::volume_dim>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, gh::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<gh::Tags::Pi<DataVector,
        gh::Solutions::WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& dt_lapse =
      get<typename WrappedGr<SolutionType>::TimeDerivLapse>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrappedGr<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift =
      get<gr::Tags::Shift<DataVector,
                          gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
          intermediate_vars);
  const auto& dt_shift =
      get<typename WrappedGr<SolutionType>::TimeDerivShift>(intermediate_vars);
  const auto& deriv_shift =
      get<typename WrappedGr<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, gh::Solutions::WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& dt_spatial_metric =
      get<typename WrappedGr<SolutionType>::TimeDerivSpatialMetric>(
          intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrappedGr<SolutionType>::DerivSpatialMetric>(
          intermediate_vars);

  const auto phi =
      gh::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);

  return {gh::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi)};
}

template <typename SolutionType>
void WrappedGr<SolutionType>::pup(PUP::er& p) {
  InitialData::pup(p);
  SolutionType::pup(p);
}

template <typename SolutionType>
PUP::able::PUP_ID WrappedGr<SolutionType>::my_PUP_ID = 0;

template <typename SolutionType>
bool operator==(const WrappedGr<SolutionType>& lhs,
                const WrappedGr<SolutionType>& rhs) {
  return static_cast<const SolutionType&>(lhs) ==
         static_cast<const SolutionType&>(rhs);
}

template <typename SolutionType>
bool operator!=(const WrappedGr<SolutionType>& lhs,
                const WrappedGr<SolutionType>& rhs) {
  return not(lhs == rhs);
}

#define WRAPPED_GR_SOLUTION_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define WRAPPED_GR_INSTANTIATE(_, data)                      \
  template class gh::Solutions::WrappedGr<  \
      WRAPPED_GR_SOLUTION_TYPE(data)>;                       \
  template bool gh::Solutions::operator==(  \
      const WrappedGr<WRAPPED_GR_SOLUTION_TYPE(data)>& lhs,  \
      const WrappedGr<WRAPPED_GR_SOLUTION_TYPE(data)>& rhs); \
  template bool gh::Solutions::operator!=(  \
      const WrappedGr<WRAPPED_GR_SOLUTION_TYPE(data)>& lhs,  \
      const WrappedGr<WRAPPED_GR_SOLUTION_TYPE(data)>& rhs);
}  // namespace gh::Solutions
