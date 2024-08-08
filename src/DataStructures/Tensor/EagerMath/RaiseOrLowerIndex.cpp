// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename X, typename Symm, typename IndexList>
class Tensor;

template <typename DataType, typename Index0, typename Index1>
void raise_or_lower_first_index(
    const gsl::not_null<
        Tensor<DataType, Symmetry<2, 1, 1>,
               index_list<change_index_up_lo<Index0>, Index1, Index1>>*>
        result,
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      for (size_t k = j; k < dimension; ++k) {
        result->get(i, j, k) = tensor.get(0, j, k) * metric.get(i, 0);
        for (size_t m = 1; m < dimension; ++m) {
          result->get(i, j, k) += tensor.get(m, j, k) * metric.get(i, m);
        }
      }
    }
  }
}

template <typename DataType, typename Index0>
void raise_or_lower_index(
    const gsl::not_null<
        Tensor<DataType, Symmetry<1>, index_list<change_index_up_lo<Index0>>>*>
        result,
    const Tensor<DataType, Symmetry<1>, index_list<Index0>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) {
  constexpr auto dimension = Index0::dim;

  for (size_t i = 0; i < dimension; ++i) {
    result->get(i) = get<0>(tensor) * metric.get(i, 0);
    for (size_t m = 1; m < dimension; ++m) {
      result->get(i) += tensor.get(m) * metric.get(i, m);
    }
  }
}

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

#define INSTANTIATE(_, data)                                                  \
  template void raise_or_lower_first_index(                                   \
      const gsl::not_null<Tensor<DTYPE(data), Symmetry<2, 1, 1>,              \
                                 index_list<change_index_up_lo<INDEX0(data)>, \
                                            INDEX1(data), INDEX1(data)>>*>    \
          result,                                                             \
      const Tensor<DTYPE(data), Symmetry<2, 1, 1>,                            \
                   index_list<INDEX0(data), INDEX1(data), INDEX1(data)>>&     \
          tensor,                                                             \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                               \
                   index_list<change_index_up_lo<INDEX0(data)>,               \
                              change_index_up_lo<INDEX0(data)>>>& metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Distorted>,
                         Frame::Spherical<Frame::Grid>),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up),
                        (UpLo::Lo, UpLo::Up))

#undef FRAME1
#undef UPORLO1
#undef INDEXTYPE1
#undef INDEX1
#undef INSTANTIATE

#define INSTANTIATE2(_, data)                                           \
  template void raise_or_lower_index(                                   \
      const gsl::not_null<                                              \
          Tensor<DTYPE(data), Symmetry<1>,                              \
                 index_list<change_index_up_lo<INDEX0(data)>>>*>        \
          trace_of_tensor,                                              \
      const Tensor<DTYPE(data), Symmetry<1>, index_list<INDEX0(data)>>& \
          tensor,                                                       \
      const Tensor<DTYPE(data), Symmetry<1, 1>,                         \
                   index_list<change_index_up_lo<INDEX0(data)>,         \
                              change_index_up_lo<INDEX0(data)>>>& metric);

GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (SpatialIndex, SpacetimeIndex), (UpLo::Lo, UpLo::Up))

#undef DIM
#undef DTYPE
#undef FRAME0
#undef UPORLO0
#undef INDEXTYPE0
#undef INDEX0
#undef INSTANTIATE2
