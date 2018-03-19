// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/Minkowski.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace EinsteinSolutions {

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {Scalar<DataType>(make_with_value<DataType>(x, 1.))};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::dt<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        Tags::dt<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>>> /*meta*/)
    const noexcept {
  return {Scalar<DataType>(make_with_value<DataType>(x, 0.))};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::deriv<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>,
                                tmpl::size_t<Dim>, Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<Tags::deriv<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::Shift<Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                                tmpl::size_t<Dim>, Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::iJ<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  tnsr::ii<DataType, Dim> lower_metric(make_with_value<DataType>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    lower_metric.get(i, i) = 1.;
  }
  return {std::move(lower_metric)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<Tags::dt<gr::Tags::SpatialMetric<
                              Dim, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::ii<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                tmpl::size_t<Dim>, Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::ijj<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::InverseSpatialMetric<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  tnsr::II<DataType, Dim> upper_metric(make_with_value<DataType>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    upper_metric.get(i, i) = 1.;
  }
  return {std::move(upper_metric)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::ExtrinsicCurvature<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::ii<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::SqrtDetSpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::SqrtDetSpatialMetric<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    Tags::dt<gr::Tags::SqrtDetSpatialMetric<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<Tags::dt<gr::Tags::SqrtDetSpatialMetric<
                              Dim, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}
} // namespace EinsteinSolutions

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>>                \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<                                                              \
          gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Tags::dt<gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>>>      \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::dt<                                                     \
          gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Tags::deriv<gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>,    \
                  tmpl::size_t<DIM(data)>, Frame::Inertial>>                   \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::deriv<                                                  \
          gr::Tags::Lapse<DIM(data), Frame::Inertial, DTYPE(data)>,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>                \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<                                                              \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Tags::dt<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>>      \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::dt<                                                     \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Tags::deriv<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,    \
                  tmpl::size_t<DIM(data)>, Frame::Inertial>>                   \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::deriv<                                                  \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>        \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SpatialMetric<DIM(data), Frame::Inertial,           \
                                         DTYPE(data)>> /*meta*/)               \
      const noexcept;                                                          \
  template tuples::TaggedTuple<Tags::dt<                                       \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>>       \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::dt<gr::Tags::SpatialMetric<DIM(data), Frame::Inertial,  \
                                                  DTYPE(data)>>> /*meta*/)     \
      const noexcept;                                                          \
  template tuples::TaggedTuple<Tags::deriv<                                    \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,        \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>                               \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::deriv<                                                  \
          gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,    \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>> \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial,    \
                                                DTYPE(data)>> /*meta*/)        \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial, DTYPE(data)>>   \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial,      \
                                              DTYPE(data)>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SqrtDetSpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>> \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DIM(data), Frame::Inertial,    \
                                                DTYPE(data)>> /*meta*/)        \
      const noexcept;                                                          \
  template tuples::TaggedTuple<Tags::dt<gr::Tags::SqrtDetSpatialMetric<        \
      DIM(data), Frame::Inertial, DTYPE(data)>>>                               \
  EinsteinSolutions::Minkowski<DIM(data)>::variables(                          \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<Tags::dt<gr::Tags::SqrtDetSpatialMetric<                      \
          DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
