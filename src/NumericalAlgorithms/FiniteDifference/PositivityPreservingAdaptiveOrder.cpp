// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"

#include <array>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/FiniteDifference/Minmod.hpp"
#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.tpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace fd::reconstruction {
namespace {
// use type alias for the function pointer return type for brevity
template <size_t Dim>
using pointer_return_type =
    std::tuple<void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                        gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                        const gsl::span<const double>&,
                        const DirectionMap<Dim, gsl::span<const double>>&,
                        const Index<Dim>&, size_t, double),
               void (*)(gsl::not_null<DataVector*>, const DataVector&,
                        const DataVector&, const Index<Dim>&, const Index<Dim>&,
                        const Direction<Dim>&, const double&),
               void (*)(gsl::not_null<DataVector*>, const DataVector&,
                        const DataVector&, const Index<Dim>&, const Index<Dim>&,
                        const Direction<Dim>&, const double&)>;

template <typename LowOrderReconstructor, bool PositivityPreserving, size_t Dim>
pointer_return_type<Dim> function_pointer() {
  return {&positivity_preserving_adaptive_order<LowOrderReconstructor,
                                                PositivityPreserving, Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Lower,
              ::fd::reconstruction::detail::
                  PositivityPreservingAdaptiveOrderReconstructor<
                      LowOrderReconstructor, PositivityPreserving>,
              Dim>,
          &::fd::reconstruction::reconstruct_neighbor<
              Side::Upper,
              ::fd::reconstruction::detail::
                  PositivityPreservingAdaptiveOrderReconstructor<
                      LowOrderReconstructor, PositivityPreserving>,
              Dim>};
}

template <bool PositivityPreserving, size_t Dim>
pointer_return_type<Dim>
positivity_preserving_adaptive_order_function_pointers_select_recons(
    const FallbackReconstructorType fallback_recons) {
  switch (fallback_recons) {
    case FallbackReconstructorType::Minmod:
      return function_pointer<detail::MinmodReconstructor, PositivityPreserving,
                              Dim>();
    case FallbackReconstructorType::MonotonisedCentral:
      return function_pointer<detail::MonotonisedCentralReconstructor,
                              PositivityPreserving, Dim>();
    case FallbackReconstructorType::None:
      ERROR(
          "Can't have None as the low-order reconstructor in "
          "positivity_preserving_adaptive_order.");
    default:  // LCOV_EXCL_LINE
              // LCOV_EXCL_START
      ERROR("Unsupported type of fallback reconstruction : " << fallback_recons
                                                             << "\n");
      // LCOV_EXCL_STOP
  }
}

template <size_t Dim>
pointer_return_type<Dim>
positivity_preserving_adaptive_order_function_pointers_select_positivity(
    const bool positivity_preserving,
    const FallbackReconstructorType fallback_recons) {
  if (positivity_preserving) {
    return positivity_preserving_adaptive_order_function_pointers_select_recons<
        true, Dim>(fallback_recons);
  }
  return positivity_preserving_adaptive_order_function_pointers_select_recons<
      false, Dim>(fallback_recons);
}
}  // namespace

template <size_t Dim>
pointer_return_type<Dim> positivity_preserving_adaptive_order_function_pointers(
    const bool positivity_preserving,
    const FallbackReconstructorType fallback_recons) {
  return
      positivity_preserving_adaptive_order_function_pointers_select_positivity<
      Dim>(positivity_preserving, fallback_recons);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                            \
  template pointer_return_type<DIM(data)>                 \
  positivity_preserving_adaptive_order_function_pointers( \
      bool positivity_preserving, FallbackReconstructorType fallback_recons);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define POSITIVITY_PRESERVING(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FALLBACK_RECONSTRUCTOR(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct<PositivityPreservingAdaptiveOrderReconstructor<    \
      FALLBACK_RECONSTRUCTOR(data), POSITIVITY_PRESERVING(data)>>(             \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_upper_side_of_face_vars,                               \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*>                 \
          reconstructed_lower_side_of_face_vars,                               \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Index<DIM(data)>& volume_extents,                                  \
      const size_t number_of_variables, const double& four_to_the_alpha_5);

namespace detail {
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false),
                        (MinmodReconstructor, MonotonisedCentralReconstructor))
}  // namespace detail

#undef INSTANTIATION

#define SIDE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(r, data)                                                 \
  template void reconstruct_neighbor<                                          \
      SIDE(data),                                                              \
      detail::PositivityPreservingAdaptiveOrderReconstructor<                  \
          FALLBACK_RECONSTRUCTOR(data), POSITIVITY_PRESERVING(data)>>(         \
      gsl::not_null<DataVector*> face_data, const DataVector& volume_data,     \
      const DataVector& neighbor_data, const Index<DIM(data)>& volume_extents, \
      const Index<DIM(data)>& ghost_data_extents,                              \
      const Direction<DIM(data)>& direction_to_reconstruct,                    \
      const double& four_to_the_alpha_5);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false),
                        (detail::MinmodReconstructor,
                         detail::MonotonisedCentralReconstructor),
                        (Side::Upper, Side::Lower))

#undef INSTANTIATION
#undef SIDE
#undef NONLINEAR_WEIGHT_EXPONENT
#undef FALLBACK_RECONSTRUCTOR
#undef DIM

}  // namespace fd::reconstruction
