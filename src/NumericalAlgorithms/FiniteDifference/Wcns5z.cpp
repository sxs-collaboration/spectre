// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"

#include <array>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace fd::reconstruction {

template <size_t Dim>
std::tuple<void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                    gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                    const gsl::span<const double>&,
                    const DirectionMap<Dim, gsl::span<const double>>&,
                    const Index<Dim>&, size_t, double, size_t),
           void (*)(gsl::not_null<DataVector*>, const DataVector&,
                    const DataVector&, const Index<Dim>&, const Index<Dim>&,
                    const Direction<Dim>&, const double&, const size_t&),
           void (*)(gsl::not_null<DataVector*>, const DataVector&,
                    const DataVector&, const Index<Dim>&, const Index<Dim>&,
                    const Direction<Dim>&, const double&, const size_t&)>
wcns5z_function_pointers(const size_t nonlinear_weight_exponent,
                         const bool use_fallback_to_monotonised_central) {
  switch (nonlinear_weight_exponent) {
    case 1:
      if (use_fallback_to_monotonised_central) {
        return {&wcns5z<1, true, Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Lower,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<1, true>,
                    Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Upper,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<1, true>,
                    Dim>};
      } else {
        return {&wcns5z<1, false, Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Lower,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<1, false>,
                    Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Upper,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<1, false>,
                    Dim>};
      }
    case 2:
      if (use_fallback_to_monotonised_central) {
        return {&wcns5z<2, true, Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Lower,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<2, true>,
                    Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Upper,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<2, true>,
                    Dim>};
      } else {
        return {&wcns5z<2, false, Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Lower,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<2, false>,
                    Dim>,
                &::fd::reconstruction::reconstruct_neighbor<
                    Side::Upper,
                    ::fd::reconstruction::detail::Wcns5zReconstructor<2, false>,
                    Dim>};
      }
    default:
      ERROR("Nonlinear weight exponent should be 1 or 2 but is : "
            << nonlinear_weight_exponent << "\n");
  };
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template std::tuple<                                                   \
      void (*)(gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>, \
               gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>, \
               const gsl::span<const double>&,                           \
               const DirectionMap<DIM(data), gsl::span<const double>>&,  \
               const Index<DIM(data)>&, size_t, double, size_t),         \
      void (*)(gsl::not_null<DataVector*>, const DataVector&,            \
               const DataVector&, const Index<DIM(data)>&,               \
               const Index<DIM(data)>&, const Direction<DIM(data)>&,     \
               const double&, const size_t&),                            \
      void (*)(gsl::not_null<DataVector*>, const DataVector&,            \
               const DataVector&, const Index<DIM(data)>&,               \
               const Index<DIM(data)>&, const Direction<DIM(data)>&,     \
               const double&, const size_t&)>                            \
  wcns5z_function_pointers(const size_t nonlinear_weight_exponent,       \
                           const bool use_fallback_to_monotonised_central);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define NONLINEAR_WEIGHT_EXPONENT(data) BOOST_PP_TUPLE_ELEM(1, data)
#define USE_FALLBACK_TO_MONOTONISED_CENTRAL(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                 \
  template void                                                                \
  reconstruct<Wcns5zReconstructor<NONLINEAR_WEIGHT_EXPONENT(data),             \
                                  USE_FALLBACK_TO_MONOTONISED_CENTRAL(data)>>( \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_upper_side_of_face_vars,                               \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_lower_side_of_face_vars,                               \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents,                                  \
      const size_t number_of_variables, const double& epsilon,                 \
      const size_t& max_number_of_extrema);

namespace detail {
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2), (false, true))
}  // namespace detail

#undef INSTANTIATION

#define SIDE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct_neighbor<                                          \
      SIDE(data),                                                              \
      detail::Wcns5zReconstructor<NONLINEAR_WEIGHT_EXPONENT(data),             \
                                  USE_FALLBACK_TO_MONOTONISED_CENTRAL(data)>>( \
      gsl::not_null<DataVector*> face_data, const DataVector& volume_data,     \
      const DataVector& neighbor_data, const Index<DIM(data)>& volume_extents, \
      const Index<DIM(data)>& ghost_data_extents,                              \
      const Direction<DIM(data)>& direction_to_reconstruct,                    \
      const double& epsilon, const size_t& max_number_of_extrema);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2), (false, true),
                        (Side::Upper, Side::Lower))

#undef INSTANTIATION
#undef SIDE
#undef NONLINEAR_WEIGHT_EXPONENT
#undef USE_FALLBACK_TO_MONOTONISED_CENTRAL
#undef DIM

}  // namespace fd::reconstruction
