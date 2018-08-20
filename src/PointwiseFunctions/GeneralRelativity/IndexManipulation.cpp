// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename X, typename Symm, typename IndexList>
class Tensor;

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

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<1>, index_list<Index0>> trace_last_indices(
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index1>,
                            change_index_up_lo<Index1>>>& metric) noexcept {
  auto trace_of_tensor =
      make_with_value<Tensor<DataType, Symmetry<1>, index_list<Index0>>>(metric,
                                                                         0.);
  constexpr auto dimension_of_trace = Index0::dim;
  constexpr auto dimension_of_metric = Index1::dim;
  for (size_t i = 0; i < dimension_of_trace; ++i) {
    for (size_t j = 0; j < dimension_of_metric; ++j) {
      trace_of_tensor.get(i) += tensor.get(i, j, j) * metric.get(j, j);
      for (size_t k = j + 1; k < dimension_of_metric; ++k) {
        trace_of_tensor.get(i) += 2.0 * tensor.get(i, j, k) * metric.get(j, k);
      }
    }
  }
  return trace_of_tensor;
}

template <typename DataType, typename Index0>
Scalar<DataType> trace(
    const Tensor<DataType, Symmetry<1, 1>, index_list<Index0, Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  auto trace = make_with_value<Scalar<DataType>>(tensor, 0.0);
  constexpr auto dimension = Index0::dim;
  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = i + 1; j < dimension; ++j) {  // symmetry
      get(trace) += 2.0 * tensor.get(i, j) * metric.get(i, j);
    }
    get(trace) += tensor.get(i, i) * metric.get(i, i);
  }
  return trace;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)
#define UPORLO0(data) BOOST_PP_TUPLE_ELEM(4, data)
#define UPORLO1(data) BOOST_PP_TUPLE_ELEM(5, data)

// Note: currently only instantiate these functions for the cases
// where Index0 and Index1 have the same spatial dimension, Frame and
// IndexType.  The functions above are generic and can handle any of
// them being different, if these cases are needed.
#define INDEX0(data) INDEXTYPE(data)<DIM(data), UPORLO0(data), FRAME(data)>
#define INDEX1(data) INDEXTYPE(data)<DIM(data), UPORLO1(data), FRAME(data)>

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
  template Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>        \
  trace_last_indices(                                                        \
      const Tensor<DTYPE(data), Symmetry<2, 1, 1>,                           \
                   index_list<INDEX0(data), INDEX1(data), INDEX1(data)>>&    \
          tensor,                                                            \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                              \
                   index_list<change_index_up_lo<INDEX1(data)>,              \
                              change_index_up_lo<INDEX1(data)>>>&            \
          metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up),
                        (UpLo::Lo, UpLo::Up))

#undef FRAME1
#undef UPORLO1
#undef INDEXTYPE1
#undef INDEX1
#undef INSTANTIATE

#define INSTANTIATE2(_, data)                                           \
  template Tensor<DTYPE(data), Symmetry<1>,                             \
                  index_list<change_index_up_lo<INDEX0(data)>>>         \
  raise_or_lower_index(                                                 \
      const Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>& \
          tensor,                                                       \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                         \
                   index_list<change_index_up_lo<INDEX0(data)>,         \
                              change_index_up_lo<INDEX0(data)>>>&       \
          metric) noexcept;                                             \
  template Scalar<DTYPE(data)> trace(                                   \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                         \
                   index_list<INDEX0(data), INDEX0(data)>>& tensor,     \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                         \
                   index_list<change_index_up_lo<INDEX0(data)>,         \
                              change_index_up_lo<INDEX0(data)>>>&       \
          metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up))

#undef DIM
#undef DTYPE
#undef FRAME0
#undef UPORLO0
#undef INDEXTYPE0
#undef INDEX0
#undef INSTANTIATE2
/// \endcond
