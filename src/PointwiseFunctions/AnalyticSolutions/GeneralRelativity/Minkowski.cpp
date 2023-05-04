// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare ::Tags::deriv

namespace gr::Solutions {

template <size_t Dim>
PUP::able::PUP_ID Minkowski<Dim>::my_PUP_ID = 0;

template <size_t Dim>
Minkowski<Dim>::Minkowski(CkMigrateMessage* /*msg*/) {}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Lapse<DataType>> Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const {
  return {Scalar<DataType>(make_with_value<DataType>(x, 1.))};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const {
  return {Scalar<DataType>(make_with_value<DataType>(x, 0.))};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                                  Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                             Frame::Inertial>> /*meta*/) const {
  return {make_with_value<tnsr::i<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::Shift<Dim, Frame::Inertial, DataType>> /*meta*/)
    const {
  return {make_with_value<tnsr::I<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        ::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>> /*meta*/)
    const {
  return {make_with_value<tnsr::I<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                             tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/)
    const {
  return {make_with_value<tnsr::iJ<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::SpatialMetric<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
  tnsr::ii<DataType, Dim> lower_metric(make_with_value<DataType>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    lower_metric.get(i, i) = 1.;
  }
  return {std::move(lower_metric)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial,
                                                  DataType>>> /*meta*/) const {
  return {make_with_value<tnsr::ii<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<
        ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                      tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/) const {
  return {make_with_value<tnsr::ijj<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::InverseSpatialMetric<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
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
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
  return {make_with_value<tnsr::ii<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>> /*meta*/)
    const {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::DerivDetSpatialMetric<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::DerivDetSpatialMetric<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
  return {make_with_value<tnsr::i<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::SpatialChristoffelFirstKind<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::SpatialChristoffelFirstKind<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
  return {make_with_value<tnsr::ijj<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::SpatialChristoffelSecondKind<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                          tmpl::list<gr::Tags::SpatialChristoffelSecondKind<
                              Dim, Frame::Inertial, DataType>> /*meta*/) const {
  return {make_with_value<tnsr::Ijj<DataType, Dim>>(x, 0.)};
}

template <size_t Dim>
template <typename DataType>
tuples::TaggedTuple<
    gr::Tags::TraceSpatialChristoffelSecondKind<Dim, Frame::Inertial, DataType>>
Minkowski<Dim>::variables(
    const tnsr::I<DataType, Dim>& x, double /*t*/,
    tmpl::list<gr::Tags::TraceSpatialChristoffelSecondKind<
        Dim, Frame::Inertial, DataType>> /*meta*/) const {
  return {make_with_value<tnsr::I<DataType, Dim>>(x, 0.)};
}

}  // namespace gr::Solutions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const;                \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/) const;    \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const;                     \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>                \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<                                                              \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>>    \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,  \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const;          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>        \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SpatialMetric<DIM(data), Frame::Inertial,           \
                                         DTYPE(data)>> /*meta*/) const;        \
  template tuples::TaggedTuple<::Tags::dt<                                     \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>>       \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<                           \
          DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) const;          \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,        \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>                               \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,    \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const;          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>> \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial,    \
                                                DTYPE(data)>> /*meta*/) const; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial, DTYPE(data)>>   \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial,      \
                                              DTYPE(data)>> /*meta*/) const;   \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/) const; \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>>                 \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<                                                              \
          ::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::DerivDetSpatialMetric<                \
      DIM(data), Frame::Inertial, DTYPE(data)>>                                \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::DerivDetSpatialMetric<DIM(data), Frame::Inertial,   \
                                                 DTYPE(data)>> /*meta*/)       \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SpatialChristoffelFirstKind<          \
      DIM(data), Frame::Inertial, DTYPE(data)>>                                \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SpatialChristoffelFirstKind<                        \
          DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/) const;           \
  template tuples::TaggedTuple<gr::Tags::SpatialChristoffelSecondKind<         \
      DIM(data), Frame::Inertial, DTYPE(data)>>                                \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::SpatialChristoffelSecondKind<                       \
          DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/) const;           \
  template tuples::TaggedTuple<gr::Tags::TraceSpatialChristoffelSecondKind<    \
      DIM(data), Frame::Inertial, DTYPE(data)>>                                \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::TraceSpatialChristoffelSecondKind<                  \
          DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/) const;           \
  template tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>> \
  gr::Solutions::Minkowski<DIM(data)>::variables(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/,                  \
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>> /*meta*/)     \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef INSTANTIATE

#undef DTYPE

#define INSTANTIATE(_, data) template class gr::Solutions::Minkowski<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
