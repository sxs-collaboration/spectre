// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace gr::Solutions {
template <size_t Dim>
GaugePlaneWave<Dim>::GaugePlaneWave(CkMigrateMessage* /*msg*/) {}

template <size_t Dim>
GaugePlaneWave<Dim>::GaugePlaneWave(
    std::array<double, Dim> wave_vector,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> profile)
    : wave_vector_(std::move(wave_vector)),
      profile_(std::move(profile)),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
GaugePlaneWave<Dim>::GaugePlaneWave(const GaugePlaneWave& other)
    : wave_vector_(other.wave_vector_),
      profile_(other.profile_->get_clone()),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
GaugePlaneWave<Dim>& GaugePlaneWave<Dim>::operator=(
    const GaugePlaneWave& other) {
  wave_vector_ = other.wave_vector_;
  omega_ = magnitude(wave_vector_);
  profile_ = other.profile_->get_clone();
  return *this;
}

template <size_t Dim>
void GaugePlaneWave<Dim>::pup(PUP::er& p) {
  p | wave_vector_;
  p | profile_;
  p | omega_;
}

template <size_t Dim>
template <typename DataType>
GaugePlaneWave<Dim>::IntermediateVars<DataType>::IntermediateVars(
    const std::array<double, Dim>& wave_vector,
    const std::unique_ptr<MathFunction<1, Frame::Inertial>>& profile,
    double omega, const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
    double t) {
  auto u = make_with_value<DataType>(x, -omega * t);
  for (size_t d = 0; d < Dim; ++d) {
    u += gsl::at(wave_vector, d) * x.get(d);
  }
  h = profile->operator()(u);
  du_h = profile->first_deriv(u);
  det_gamma = 1.0 + h * square(omega);
  lapse = 1.0 / sqrt(det_gamma);
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  return {Scalar<DataType>{vars.lapse}};
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {Scalar<DataType>{0.5 * cube(omega_) * vars.du_h * cube(vars.lapse)}};
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  tnsr::i<DataType, volume_dim, Frame::Inertial> d_lapse{
      -0.5 * square(omega_) * vars.du_h * cube(vars.lapse)};
  for (size_t i = 0; i < volume_dim; ++i) {
    d_lapse.get(i) *= gsl::at(wave_vector_, i);
  }
  return d_lapse;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Shift<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Shift<DataType, volume_dim>> {
  tnsr::I<DataType, volume_dim, Frame::Inertial> shift{-omega_ * vars.h /
                                                       vars.det_gamma};
  for (size_t i = 0; i < volume_dim; ++i) {
    shift.get(i) *= gsl::at(wave_vector_, i);
  }
  return shift;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> /*meta*/)
    const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> {
  tnsr::I<DataType, volume_dim, Frame::Inertial> dt_shift{
      square(omega_) * vars.du_h / square(vars.det_gamma)};
  for (size_t i = 0; i < volume_dim; ++i) {
    dt_shift.get(i) *= gsl::at(wave_vector_, i);
  }
  return dt_shift;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<DerivShift<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  tnsr::iJ<DataType, volume_dim, Frame::Inertial> d_shift{
      -omega_ * vars.du_h / square(vars.det_gamma)};
  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = 0; j < volume_dim; ++j) {
      d_shift.get(i, j) *= gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j);
    }
  }
  return d_shift;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, volume_dim>> {
  tnsr::ii<DataType, Dim, Frame::Inertial> spatial_metric{vars.h};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      spatial_metric.get(i, j) *=
          gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j);
    }
    spatial_metric.get(i, i) += 1.0;
  }
  return spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> /*meta*/)
    const -> tuples::TaggedTuple<
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> {
  tnsr::ii<DataType, Dim, Frame::Inertial> dt_spatial_metric{-omega_ *
                                                             vars.du_h};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      dt_spatial_metric.get(i, j) *=
          gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j);
    }
  }
  return dt_spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  tnsr::ijj<DataType, Dim, Frame::Inertial> d_spatial_metric{vars.du_h};
  for (size_t k = 0; k < Dim; ++k) {
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {
        d_spatial_metric.get(k, i, j) *= gsl::at(wave_vector_, k) *
                                         gsl::at(wave_vector_, i) *
                                         gsl::at(wave_vector_, j);
      }
    }
  }
  return d_spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  return {Scalar<DataType>{sqrt(vars.det_gamma)}};
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> /*meta*/)
    const
    -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> {
  tnsr::ii<DataType, Dim, Frame::Inertial> extrinsic_curvature{
      -0.5 * omega_ * vars.du_h * vars.lapse};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      extrinsic_curvature.get(i, j) *=
          gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j);
    }
  }
  return extrinsic_curvature;
}

template <size_t Dim>
template <typename DataType>
auto GaugePlaneWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, volume_dim>> /*meta*/)
    const -> tuples::TaggedTuple<
        gr::Tags::InverseSpatialMetric<DataType, volume_dim>> {
  tnsr::II<DataType, Dim, Frame::Inertial> inv_spatial_metric{vars.h /
                                                              vars.det_gamma};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inv_spatial_metric.get(i, j) *=
          -gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j);
    }
    inv_spatial_metric.get(i, i) += 1.0;
  }
  return inv_spatial_metric;
}

template <size_t Dim>
bool operator==(const GaugePlaneWave<Dim>& lhs,
                const GaugePlaneWave<Dim>& rhs) {
  return lhs.wave_vector_ == rhs.wave_vector_ and
         *(lhs.profile_) == *(rhs.profile_) and lhs.omega_ == rhs.omega_;
}

template <size_t Dim>
bool operator!=(const GaugePlaneWave<Dim>& lhs,
                const GaugePlaneWave<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>::          \
      IntermediateVars(                                                        \
          const std::array<double, DIM(data)>& wave_vector,                    \
          const std::unique_ptr<MathFunction<1, Frame::Inertial>>& profile,    \
          double omega,                                                        \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,           \
          const double t);                                                     \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& /*x*/,           \
      const double /*t*/,                                                      \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const;                \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/) const;    \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const;                     \
  template tuples::TaggedTuple<gr::Tags::Shift<DTYPE(data), DIM(data)>>        \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<gr::Tags::Shift<DTYPE(data), DIM(data)>> /*meta*/) const;     \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>>                     \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>> /*meta*/)       \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,                   \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,               \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>                         \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>>             \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>> /*meta*/) const;   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,           \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,       \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>>                  \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>>                    \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<                                                              \
          gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>> /*meta*/)      \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  GaugePlaneWave<DIM(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugePlaneWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,    \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef INSTANTIATE
#define INSTANTIATE(_, data)                                      \
  template class GaugePlaneWave<DIM(data)>;                       \
  template bool operator==(const GaugePlaneWave<DIM(data)>& lhs,  \
                           const GaugePlaneWave<DIM(data)>& rhs); \
  template bool operator!=(const GaugePlaneWave<DIM(data)>& lhs,  \
                           const GaugePlaneWave<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
}  // namespace gr::Solutions
