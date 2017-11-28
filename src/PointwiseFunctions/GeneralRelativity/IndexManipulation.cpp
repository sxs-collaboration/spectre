// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename DataType, typename Index0, typename Index1>
Tensor<DataType, Symmetry<2, 1, 1>,
       index_list<change_index_up_lo<Index0>, Index1, Index1>>
 raise_or_lower_first_index (
    const Tensor<DataType, Symmetry<2, 1, 1>,
                 index_list<Index0, Index1, Index1>>& tensor,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index0>,
                            change_index_up_lo<Index0>>>& metric) noexcept {
  Tensor<DataType, Symmetry<2, 1, 1>,
         index_list<change_index_up_lo<Index0>, Index1, Index1>>
      tensor_opposite_valence{make_with_value<DataType>(*tensor.begin(), 0.)};

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

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define UPORLO(data) BOOST_PP_TUPLE_ELEM(3, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(4, data)

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
          metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial), (UpLo::Lo, UpLo::Up),
                        (SpatialIndex, SpacetimeIndex))

#undef DIM
#undef DTYPE
#undef FRAME
#undef UPORLO
#undef INDEXTYPE
#undef INDEX0
#undef INDEX1
#undef INSTANTIATE
/// \endcond
