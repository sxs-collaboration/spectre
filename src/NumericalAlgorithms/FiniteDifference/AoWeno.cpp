// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/AoWeno.hpp"

#include <array>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace fd::reconstruction {
template <size_t Dim>
std::tuple<
    void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             gsl::not_null<std::array<gsl::span<double>, Dim>*>,
             const gsl::span<const double>&,
             const DirectionMap<Dim, gsl::span<const double>>&,
             const Index<Dim>&, size_t, double, double, double),
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&, const double&, const double&),
    void (*)(gsl::not_null<DataVector*>, const DataVector&, const DataVector&,
             const Index<Dim>&, const Index<Dim>&, const Direction<Dim>&,
             const double&, const double&, const double&)>
aoweno_53_function_pointers(const size_t nonlinear_weight_exponent) {
  switch (nonlinear_weight_exponent) {
    case 2:
      return {&aoweno_53<2, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Lower,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<2>, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Upper,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<2>, Dim>};
    case 4:
      return {&aoweno_53<4, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Lower,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<4>, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Upper,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<4>, Dim>};
    case 6:
      return {&aoweno_53<6, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Lower,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<6>, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Upper,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<6>, Dim>};
    case 8:
      return {&aoweno_53<8, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Lower,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<8>, Dim>,
              &::fd::reconstruction::reconstruct_neighbor<
                  Side::Upper,
                  ::fd::reconstruction::detail::AoWeno53Reconstructor<8>, Dim>};
    case 10:
      return {
          &aoweno_53<10, Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Lower,
              ::fd::reconstruction::detail::AoWeno53Reconstructor<10>, Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Upper,
              ::fd::reconstruction::detail::AoWeno53Reconstructor<10>, Dim>};
    case 12:
      return {
          &aoweno_53<12, Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Lower,
              ::fd::reconstruction::detail::AoWeno53Reconstructor<12>, Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Upper,
              ::fd::reconstruction::detail::AoWeno53Reconstructor<12>, Dim>};
    default:
      ERROR("Unsupported nonlinear weight exponent "
            << nonlinear_weight_exponent << " only have 2,4,6,8,10,12 allowed");
  };
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template std::tuple<                                                   \
      void (*)(gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>, \
               gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>, \
               const gsl::span<const double>&,                           \
               const DirectionMap<DIM(data), gsl::span<const double>>&,  \
               const Index<DIM(data)>&, size_t, double, double, double), \
      void (*)(gsl::not_null<DataVector*>, const DataVector&,            \
               const DataVector&, const Index<DIM(data)>&,               \
               const Index<DIM(data)>&, const Direction<DIM(data)>&,     \
               const double&, const double&, const double&),             \
      void (*)(gsl::not_null<DataVector*>, const DataVector&,            \
               const DataVector&, const Index<DIM(data)>&,               \
               const Index<DIM(data)>&, const Direction<DIM(data)>&,     \
               const double&, const double&, const double&)>             \
  aoweno_53_function_pointers(const size_t nonlinear_weight_exponent);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define NONLINEAR_WEIGHT_EXPONENT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                 \
  template void                                                                \
  reconstruct<AoWeno53Reconstructor<NONLINEAR_WEIGHT_EXPONENT(data)>>(         \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_upper_side_of_face_vars,                               \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_lower_side_of_face_vars,                               \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents,                                  \
      const size_t number_of_variables, const double& gamma_hi,                \
      const double& gamma_lo, const double& epsilon);

namespace detail {
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (2, 4, 6, 8, 10, 12))
}  // namespace detail

#undef INSTANTIATION

#define SIDE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct_neighbor<                                          \
      SIDE(data),                                                              \
      detail::AoWeno53Reconstructor<NONLINEAR_WEIGHT_EXPONENT(data)>>(         \
      gsl::not_null<DataVector*> face_data, const DataVector& volume_data,     \
      const DataVector& neighbor_data, const Index<DIM(data)>& volume_extents, \
      const Index<DIM(data)>& ghost_data_extents,                              \
      const Direction<DIM(data)>& direction_to_reconstruct,                    \
      const double& gamma_hi, const double& gamma_lo, const double& epsilon);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (2, 4, 6, 8, 10, 12),
                        (Side::Upper, Side::Lower))

#undef INSTANTIATION
#undef SIDE
#undef NONLINEAR_WEIGHT_EXPONENT
#undef DIM
}  // namespace fd::reconstruction
