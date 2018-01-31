// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<2, 1, 1>,
       index_list<change_index_up_lo<Index0>, Index1, Index1>>
raise_or_lower_first_index(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  auto tensor_opposite_valence = make_with_value<
      Tensor<DataType, Symmetry<2, 1, 1>,
             index_list<change_index_up_lo<Index0>, Index1, Index1>>>(metric,
                                                                      0.);

  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      for (size_t k = j; k < dimension; ++k) {
        for (size_t m = 0; m < dimension; ++m) {
          tensor_opposite_valence.get(i, j, k) +=
              tensor.get(m, j, k) * metric.get(i, m);
        }
      }
    }
  }
  return tensor_opposite_valence;
}

template <typename DataType, typename Index0>
Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>
raise_or_lower_index(
    const Tensor<DataType, Symmetry<1>, index_list<Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  auto tensor_opposite_valence = make_with_value<
      Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>>(
      metric, 0.);

  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    for (size_t m = 0; m < dimension; ++m) {
      tensor_opposite_valence.get(i) += tensor.get(m) * metric.get(i, m);
    }
  }
  return tensor_opposite_valence;
}

template <size_t Dim, typename Frame, IndexType TypeOfIndex, typename DataType>
tnsr::a<DataType, Dim, Frame, TypeOfIndex> trace_last_indices(
    const tnsr::abb<DataType, Dim, Frame, TypeOfIndex>& tensor,
    const tnsr::AA<DataType, Dim, Frame, TypeOfIndex>& upper_metric) noexcept {
  auto trace_of_tensor =
      make_with_value<tnsr::a<DataType, Dim, Frame, TypeOfIndex>>(upper_metric,
                                                                  0.);

  const auto dimension = index_dim<0>(tensor);
  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      trace_of_tensor.get(i) += tensor.get(i, j, j) * upper_metric.get(j, j);
      for (size_t k = j + 1; k < dimension; ++k) {
        trace_of_tensor.get(i) +=
            2 * tensor.get(i, j, k) * upper_metric.get(j, k);
      }
    }
  }
  return trace_of_tensor;
}

template <size_t SpatialDim, typename Frame, IndexType TypeOfIndex,
          typename DataType>
Scalar<DataType> trace(
    const tnsr::aa<DataType, SpatialDim, Frame, TypeOfIndex>& tensor,
    const tnsr::AA<DataType, SpatialDim, Frame, TypeOfIndex>&
        upper_metric) noexcept {
  auto trace = make_with_value<Scalar<DataType>>(tensor, 0.0);
  const auto dimension = index_dim<0>(tensor);
  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = i + 1; j < dimension; ++j) {  // symmetry
      get(trace) += 2.0 * tensor.get(i, j) * upper_metric.get(i, j);
    }
    get(trace) += tensor.get(i, i) * upper_metric.get(i, i);
  }
  return trace;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)
#define UPORLO(data) BOOST_PP_TUPLE_ELEM(4, data)

#define INDEX0(data) INDEXTYPE(data)<DIM(data), UPORLO(data), FRAME(data)>
#define INDEX1(data) INDEXTYPE(data)<DIM(data), UpLo::Lo, FRAME(data)>

#define INSTANTIATE(_, data)                                                 \
  template Tensor<DTYPE(data), Symmetry<2, 1, 1>,                            \
                  index_list<change_index_up_lo<INDEX0(data)>, INDEX1(data), \
                             INDEX1(data)>>                                  \
  raise_or_lower_first_index(                                                \
      const Tensor<DTYPE(data), Symmetry<2, 1, 1>,                           \
                   index_list<INDEX0(data), INDEX1(data), INDEX1(data)>>&    \
          tensor,                                                            \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                              \
                   index_list<change_index_up_lo<INDEX0(data)>,              \
                              change_index_up_lo<INDEX0(data)>>>&            \
          metric) noexcept;                                                  \
  template Tensor<DTYPE(data), Symmetry<1>,                                  \
                  index_list<change_index_up_lo<INDEX0(data)>>>              \
  raise_or_lower_index(                                                      \
      const Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>&      \
          tensor,                                                            \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                              \
                   index_list<change_index_up_lo<INDEX0(data)>,              \
                              change_index_up_lo<INDEX0(data)>>>&            \
          metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up))

#define INSTANTIATE2(_, data)                                                \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>     \
  trace_last_indices(                                                        \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>& \
          tensor,                                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          upper_metric) noexcept;                                            \
  template Scalar<DTYPE(data)> trace(                                        \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          tensor,                                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data), INDEXTYPE(data)>&  \
          upper_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (IndexType::Spatial, IndexType::Spacetime))

#undef DIM
#undef DTYPE
#undef FRAME
#undef UPORLO
#undef INDEXTYPE
#undef INDEX0
#undef INDEX1
#undef INSTANTIATE
/// \endcond
