// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Ccz4WrappedGr.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Ccz4::Solutions {
template <typename SolutionType>
Ccz4WrappedGr<SolutionType>::Ccz4WrappedGr(const SolutionType& wrapped_solution)
    : SolutionType(wrapped_solution) {}

template <typename SolutionType>
Ccz4WrappedGr<SolutionType>::Ccz4WrappedGr(CkMigrateMessage* msg)
    : InitialData(msg), SolutionType(msg) {}

template <typename SolutionType>
std::unique_ptr<evolution::initial_data::InitialData>
Ccz4WrappedGr<SolutionType>::get_clone() const {
  return std::make_unique<Ccz4WrappedGr<SolutionType>>(*this);
}

template <typename SolutionType>
tuples::TaggedTuple<Ccz4::Tags::ConformalMetric<
    DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<Ccz4::Tags::ConformalMetric<
        DataVector,
        Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars);
  Scalar<DataVector> conformal_factor;
  get(conformal_factor) = pow(get(sqrt_det_spatial_metric), -1. / 3.);
  tnsr::ii<DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>
      conformal_spatial_metric;
  ::tenex::evaluate<ti::i, ti::j>(make_not_null(&conformal_spatial_metric),
                                  (conformal_factor)() * (conformal_factor)() *
                                      (spatial_metric)(ti::i, ti::j));
  return {std::move(conformal_spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<Ccz4::Tags::ConformalFactor<DataVector>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<Ccz4::Tags::ConformalFactor<DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars);
  Scalar<DataVector> conformal_factor;
  get(conformal_factor) = pow(get(sqrt_det_spatial_metric), -1. / 3.);

  return {std::move(conformal_factor)};
}

template <typename SolutionType>
tuples::TaggedTuple<Ccz4::Tags::ATilde<
    DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<Ccz4::Tags::ATilde<
        DataVector,
        Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars);
  Scalar<DataVector> conformal_factor_squared;
  get(conformal_factor_squared) = pow(get(sqrt_det_spatial_metric), -2. / 3.);
  const auto& extrinsic_curvature = get<gr::Tags::ExtrinsicCurvature<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& inverse_spatial_metric = get<gr::Tags::InverseSpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto trace_extrinsic_curvature =
      trace(extrinsic_curvature, inverse_spatial_metric);

  return {::Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                          extrinsic_curvature, trace_extrinsic_curvature)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataVector>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& extrinsic_curvature = get<gr::Tags::ExtrinsicCurvature<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto& inverse_spatial_metric = get<gr::Tags::InverseSpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  const auto trace_extrinsic_curvature =
      trace(extrinsic_curvature, inverse_spatial_metric);

  return {std::move(trace_extrinsic_curvature)};
}

template <typename SolutionType>
tuples::TaggedTuple<Ccz4::Tags::Theta<DataVector>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<Ccz4::Tags::Theta<DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars);
  // Theta won't be exactly zero due to numerical errors
  // in the initial data leading to constraint violations.
  // But in practice evolutions set this to zero; see e.g.
  // Dumbser2017okk
  const auto theta =
      make_with_value<Scalar<DataVector>>(sqrt_det_spatial_metric, 0.0);

  return {std::move(theta)};
}

template <typename SolutionType>
tuples::TaggedTuple<Ccz4::Tags::GammaHat<
    DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>
Ccz4WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, Ccz4::Solutions::Ccz4WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    tmpl::list<Ccz4::Tags::GammaHat<
        DataVector,
        Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>> /*meta*/,
    const IntermediateVars& intermediate_vars) const {
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars);
  Scalar<DataVector> inverse_conformal_factor_squared;
  get(inverse_conformal_factor_squared) =
      pow(get(sqrt_det_spatial_metric), 2. / 3.);
  const auto& inverse_spatial_metric = get<gr::Tags::InverseSpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  tnsr::II<DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>
      inverse_conformal_spatial_metric;
  ::tenex::evaluate<ti::I, ti::J>(
      make_not_null(&inverse_conformal_spatial_metric),
      inverse_conformal_factor_squared() *
          inverse_spatial_metric(ti::I, ti::J));

  const auto& d_spatial_metric = get<DerivSpatialMetric>(intermediate_vars);
  tnsr::i<DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>
      d_det_spatial_metric;
  ::tenex::evaluate<ti::i>(make_not_null(&d_det_spatial_metric),
                           sqrt_det_spatial_metric() *
                               sqrt_det_spatial_metric() *
                               inverse_spatial_metric(ti::J, ti::K) *
                               d_spatial_metric(ti::i, ti::j, ti::k));
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      DataVector, Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>>(
      intermediate_vars);
  tnsr::ijj<DataVector,
            Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim>
      field_d(get(inverse_conformal_factor_squared));
  for (size_t k = 0;
       k < Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim; k++) {
    for (size_t i = 0;
         i < Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim; i++) {
      for (size_t j = i;
           j < Ccz4::Solutions::Ccz4WrappedGr<SolutionType>::volume_dim; j++) {
        field_d.get(k, i, j) = d_spatial_metric.get(k, i, j) /
                                   get(inverse_conformal_factor_squared) -
                               pow(get(inverse_conformal_factor_squared), -4) *
                                   d_det_spatial_metric.get(k) *
                                   spatial_metric.get(i, j) / 3.;
        field_d.get(k, i, j) *= 0.5;
      }
    }
  }

  const auto conformal_christoffel_second_kind =
      ::Ccz4::conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d);
  // \tilde{Gamma}^i in Ccz4
  const auto contracted_conformal_christoffel_second_kind =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);

  // Similar to Theta, we assume the spatial Z4 constraints are zero,
  // so \hat{Gamma}^i = \tilde{Gamma}^i
  return {std::move(contracted_conformal_christoffel_second_kind)};
}

template <typename SolutionType>
void Ccz4WrappedGr<SolutionType>::pup(PUP::er& p) {
  InitialData::pup(p);
  SolutionType::pup(p);
}

template <typename SolutionType>
PUP::able::PUP_ID Ccz4WrappedGr<SolutionType>::my_PUP_ID = 0;

template <typename SolutionType>
bool operator==(const Ccz4WrappedGr<SolutionType>& lhs,
                const Ccz4WrappedGr<SolutionType>& rhs) {
  return static_cast<const SolutionType&>(lhs) ==
         static_cast<const SolutionType&>(rhs);
}

template <typename SolutionType>
bool operator!=(const Ccz4WrappedGr<SolutionType>& lhs,
                const Ccz4WrappedGr<SolutionType>& rhs) {
  return not(lhs == rhs);
}

#define CCZ4_WRAPPED_GR_SOLUTION_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define CCZ4_WRAPPED_GR_INSTANTIATE(_, data)                                   \
  template class Ccz4::Solutions::Ccz4WrappedGr<CCZ4_WRAPPED_GR_SOLUTION_TYPE( \
      data)>;                                                                  \
  template bool Ccz4::Solutions::operator==(                                   \
      const Ccz4WrappedGr<CCZ4_WRAPPED_GR_SOLUTION_TYPE(data)>& lhs,           \
      const Ccz4WrappedGr<CCZ4_WRAPPED_GR_SOLUTION_TYPE(data)>& rhs);          \
  template bool Ccz4::Solutions::operator!=(                                   \
      const Ccz4WrappedGr<CCZ4_WRAPPED_GR_SOLUTION_TYPE(data)>& lhs,           \
      const Ccz4WrappedGr<CCZ4_WRAPPED_GR_SOLUTION_TYPE(data)>& rhs);
}  // namespace Ccz4::Solutions
