// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/Unlimited.hpp"

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "Utilities/GenerateInstantiations.hpp"

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace fd::reconstruction {
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DEGREE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct<UnlimitedReconstructor<DEGREE(data)>>(             \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_upper_side_of_face_vars,                               \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_lower_side_of_face_vars,                               \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents,                                  \
      const size_t number_of_variables);

namespace detail {
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (2, 4))
}  // namespace detail

#undef INSTANTIATION

#define SIDE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct_neighbor<                                          \
      SIDE(data), detail::UnlimitedReconstructor<DEGREE(data)>>(               \
      gsl::not_null<DataVector*> face_data, const DataVector& volume_data,     \
      const DataVector& neighbor_data, const Index<DIM(data)>& volume_extents, \
      const Index<DIM(data)>& ghost_data_extents,                              \
      const Direction<DIM(data)>& direction_to_reconstruct);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (2, 4),
                        (Side::Upper, Side::Lower))

#undef INSTANTIATION
#undef SIDE
#undef DEGREE
#undef DIM
}  // namespace fd::reconstruction
