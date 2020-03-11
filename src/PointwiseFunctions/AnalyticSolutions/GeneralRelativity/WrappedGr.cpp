// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/GenerateInstantiations.hpp"
// IWYU pragma: no_forward_declare ::Tags::deriv

/// \cond
namespace GeneralizedHarmonic {
namespace Solutions {
// Preprocessor logic to avoid defining variables() functions for
// tags other than the three the wrapper adds (i.e., other than
// gr::Tags::SpacetimeMetric, GeneralizedHarmonic::Tags::Pi, and
// GeneralizedHarmonic:Tags::Phi)
#define FUNC_DEF(r, data, elem)                                            \
  template <typename SolutionType>                                         \
  auto WrappedGr<SolutionType>::variables(                                 \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr< \
                                    SolutionType>::volume_dim>& /*x*/,     \
      double /*t*/, tmpl::list<elem> /*meta*/,                             \
      const IntermediateVars& intermediate_vars)                           \
      const noexcept->tuples::TaggedTuple<elem> {                          \
    return {get<elem>(intermediate_vars)};                                 \
  }

#define MY_LIST                                                              \
  BOOST_PP_TUPLE_TO_LIST(                                                    \
      (gr::Tags::Lapse<DataVector>, TimeDerivLapse, DerivLapse, TagShift,    \
       TimeDerivShift, DerivShift, TagSpatialMetric, TimeDerivSpatialMetric, \
       DerivSpatialMetric, TagInverseSpatialMetric, TagExCurvature,          \
       gr::Tags::SqrtDetSpatialMetric<DataVector>))

BOOST_PP_LIST_FOR_EACH(FUNC_DEF, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DEF

template <typename SolutionType>
tuples::TaggedTuple<gr::Tags::SpacetimeMetric<
    GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
    Frame::Inertial, DataVector>>
WrappedGr<SolutionType>::variables(
    const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<
                                  SolutionType>::volume_dim>& /*x*/,
    double /*t*/,
    tmpl::list<gr::Tags::SpacetimeMetric<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial, DataVector>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
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
    double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Phi<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
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
    double /*t*/,
    tmpl::list<GeneralizedHarmonic::Tags::Pi<
        GeneralizedHarmonic::Solutions::WrappedGr<SolutionType>::volume_dim,
        Frame::Inertial>> /*meta*/,
    const IntermediateVars& intermediate_vars) const noexcept {
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
}  // namespace Solutions
}  // namespace GeneralizedHarmonic

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<gr::Tags::Lapse<DataVector>>                    \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/, tmpl::list<gr::Tags::Lapse<DataVector>> /*meta*/,          \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::       \
                                   WrappedGr<STYPE(data)>::TimeDerivLapse>     \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::TimeDerivLapse> /*meta*/,                                    \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::       \
                                   WrappedGr<STYPE(data)>::DerivLapse>         \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::DerivLapse> /*meta*/,                                        \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::Shift<                                \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial, DataVector>>                                            \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::Shift<                                              \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial, DataVector>> /*meta*/,                              \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::       \
                                   WrappedGr<STYPE(data)>::TimeDerivShift>     \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::TimeDerivShift> /*meta*/,                                    \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::       \
                                   WrappedGr<STYPE(data)>::DerivShift>         \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::DerivShift> /*meta*/,                                        \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::SpatialMetric<                        \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial, DataVector>>                                            \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::SpatialMetric<                                      \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial, DataVector>> /*meta*/,                              \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<                                                \
      typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(                \
          data)>::TimeDerivSpatialMetric>                                      \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::TimeDerivSpatialMetric> /*meta*/,                            \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<typename GeneralizedHarmonic::Solutions::       \
                                   WrappedGr<STYPE(data)>::DerivSpatialMetric> \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<typename GeneralizedHarmonic::Solutions::WrappedGr<STYPE(     \
          data)>::DerivSpatialMetric> /*meta*/,                                \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<                 \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial, DataVector>>                                            \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::InverseSpatialMetric<                               \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial, DataVector>> /*meta*/,                              \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<                   \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial, DataVector>>                                            \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::ExtrinsicCurvature<                                 \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial, DataVector>> /*meta*/,                              \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataVector>>     \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>> /*meta*/,         \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<gr::Tags::SpacetimeMetric<                      \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial, DataVector>>                                            \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<gr::Tags::SpacetimeMetric<                                    \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial, DataVector>> /*meta*/,                              \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<GeneralizedHarmonic::Tags::Pi<                  \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial>>                                                        \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<GeneralizedHarmonic::Tags::Pi<                                \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial>> /*meta*/,                                          \
      const IntermediateVars& intermediate_vars) const noexcept;               \
  template tuples::TaggedTuple<GeneralizedHarmonic::Tags::Phi<                 \
      GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,      \
      Frame::Inertial>>                                                        \
  GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::variables(           \
      const tnsr::I<DataVector, GeneralizedHarmonic::Solutions::WrappedGr<     \
                                    STYPE(data)>::volume_dim>& /*x*/,          \
      double /*t*/,                                                            \
      tmpl::list<GeneralizedHarmonic::Tags::Phi<                               \
          GeneralizedHarmonic::Solutions::WrappedGr<STYPE(data)>::volume_dim,  \
          Frame::Inertial>> /*meta*/,                                          \
      const IntermediateVars& intermediate_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::GaugeWave<1>,
                                      gr::Solutions::GaugeWave<2>,
                                      gr::Solutions::GaugeWave<3>,
                                      gr::Solutions::Minkowski<1>,
                                      gr::Solutions::Minkowski<2>,
                                      gr::Solutions::Minkowski<3>,
                                      gr::Solutions::KerrSchild))

#undef DIM
#undef STYPE
#undef INSTANTIATE
/// \endcond
