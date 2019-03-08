// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrapGh.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare ::Tags::deriv

/// \cond
namespace GeneralizedHarmonic {
namespace Solutions {
template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::Lapse<DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/, tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::Lapse<DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::TimeDerivLapse>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::TimeDerivLapse> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {
      get<typename WrapGh<SolutionType>::TimeDerivLapse>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::DerivLapse>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::DerivLapse> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<typename WrapGh<SolutionType>::DerivLapse>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::Shift<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::Shift<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::TimeDerivShift>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::TimeDerivShift> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {
      get<typename WrapGh<SolutionType>::TimeDerivShift>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::DerivShift>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::DerivShift> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<DerivShift>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::SpatialMetric<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::TimeDerivSpatialMetric>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::TimeDerivSpatialMetric> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<typename WrapGh<SolutionType>::TimeDerivSpatialMetric>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<typename WrapGh<SolutionType>::DerivSpatialMetric>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<typename WrapGh<SolutionType>::DerivSpatialMetric> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<typename WrapGh<SolutionType>::DerivSpatialMetric>(
      intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::InverseSpatialMetric<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::InverseSpatialMetric<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::ExtrinsicCurvature<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::ExtrinsicCurvature<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  return {get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(intermediate_vars)};
}

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpacetimeMetric<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::SpacetimeMetric<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);

  return {gr::spacetime_metric(lapse, shift, spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Phi<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrapGh<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& deriv_shift =
      get<typename WrapGh<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrapGh<SolutionType>::DerivSpatialMetric>(intermediate_vars);

  return {GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric)};
}

template <typename SolutionType>
tuples::TaggedTuple<GeneralizedHarmonic::Tags::Pi<
    GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
    Frame::Inertial>>
WrapGh<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Pi<
        GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(intermediate_vars);
  const auto& dt_lapse =
      get<typename WrapGh<SolutionType>::TimeDerivLapse>(intermediate_vars);
  const auto& deriv_lapse =
      get<typename WrapGh<SolutionType>::DerivLapse>(intermediate_vars);

  const auto& shift = get<gr::Tags::Shift<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& dt_shift =
      get<typename WrapGh<SolutionType>::TimeDerivShift>(intermediate_vars);
  const auto& deriv_shift =
      get<typename WrapGh<SolutionType>::DerivShift>(intermediate_vars);

  const auto& spatial_metric = get<gr::Tags::SpatialMetric<
      GeneralizedHarmonic::Solutions::WrapGh<SolutionType>::volume_dim,
      Frame::Inertial, DataVector>>(intermediate_vars);
  const auto& dt_spatial_metric =
      get<typename WrapGh<SolutionType>::TimeDerivSpatialMetric>(
          intermediate_vars);
  const auto& deriv_spatial_metric =
      get<typename WrapGh<SolutionType>::DerivSpatialMetric>(intermediate_vars);

  const auto& phi =
      GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);

  return {GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift,
                                  spatial_metric, dt_spatial_metric, phi)};
}
}  // namespace Solutions
}  // namespace GeneralizedHarmonic

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tuples::TaggedTuple<gr::Tags::Lapse<DataVector>>                   \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/, tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,         \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::      \
                                   WrapGh<STYPE(data)>::TimeDerivLapse>       \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::TimeDerivLapse> /*meta*/,                                   \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::      \
                                   WrapGh<STYPE(data)>::DerivLapse>           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::DerivLapse> /*meta*/,                                       \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::Shift<                               \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial, DataVector>>                                           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::Shift<                                             \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial, DataVector>> /*meta*/,                             \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::      \
                                   WrapGh<STYPE(data)>::TimeDerivShift>       \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::TimeDerivShift> /*meta*/,                                   \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::      \
                                   WrapGh<STYPE(data)>::DerivShift>           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::DerivShift> /*meta*/,                                       \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::SpatialMetric<                       \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial, DataVector>>                                           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::SpatialMetric<                                     \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial, DataVector>> /*meta*/,                             \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<                                               \
      typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(                  \
          data)>::TimeDerivSpatialMetric>                                     \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::TimeDerivSpatialMetric> /*meta*/,                           \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::      \
                                   WrapGh<STYPE(data)>::DerivSpatialMetric>   \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrapGh<STYPE(       \
          data)>::DerivSpatialMetric> /*meta*/,                               \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<                \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial, DataVector>>                                           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::InverseSpatialMetric<                              \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial, DataVector>> /*meta*/,                             \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<                  \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial, DataVector>>                                           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::ExtrinsicCurvature<                                \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial, DataVector>> /*meta*/,                             \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>>    \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,        \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<gr::Tags::SpacetimeMetric<                     \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial, DataVector>>                                           \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<gr::Tags::SpacetimeMetric<                                   \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial, DataVector>> /*meta*/,                             \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<GeneralizedHarmonic::Tags::Pi<                 \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial>>                                                       \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<GeneralizedHarmonic::Tags::Pi<                               \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial>> /*meta*/,                                         \
      const IntermediateVars& intermediate_vars) const noexcept;              \
  template tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<                \
      GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,        \
      Frame::Inertial>>                                                       \
  GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::variables(             \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrapGh<STYPE( \
                                    data)>::volume_dim>& /*x*/,               \
      double /*t*/,                                                           \
      tmpl::list<GeneralizedHarmonic::Tags::Phi<                              \
          GeneralizedHarmonic::Solutions::WrapGh<STYPE(data)>::volume_dim,    \
          Frame::Inertial>> /*meta*/,                                         \
      const IntermediateVars& intermediate_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::Minkowski<1>,
                                      gr::Solutions::Minkowski<2>,
                                      gr::Solutions::Minkowski<3>,
                                      gr::Solutions::KerrSchild))

#undef DIM
#undef STYPE
#undef INSTANTIATE
/// \endcond
