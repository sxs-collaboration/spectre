// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

#include "NumericalAlgorithms/FiniteDifference/MonotonisedCentral.hpp"
#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "NumericalAlgorithms/FiniteDifference/Wcns5z.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {

template <size_t NonlinearWeightExponent>
struct Wcns5zMcReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int stride,
      const size_t max_number_of_extrema, const double epsilon) {
    // count the number of extrema in the given FD stencil
    size_t n_extrema{0};
    for (int i = -1; i < 2; ++i) {
      // check if q[i * stride] is local maximum
      n_extrema += (q[i * stride] > q[(i - 1) * stride]) and
                   (q[i * stride] > q[(i + 1) * stride]);
      // check if q[i * stride] is local minimum
      n_extrema += (q[i * stride] < q[(i - 1) * stride]) and
                   (q[i * stride] < q[(i + 1) * stride]);
    }

    // if `n_extrema` is equal or smaller than a specified number, use Wcns5z
    // reconstruction
    if (n_extrema < max_number_of_extrema + 1) {
      return Wcns5zReconstructor<NonlinearWeightExponent>::pointwise(q, stride,
                                                                     epsilon);
    } else {
      // otherwise use MC reconstruction
      return MonotonisedCentralReconstructor::pointwise(q, stride);
    }
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 5; }
};

}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs adaptive reconstruction using WCNS-5Z and MC schemes.
 *
 * For each finite difference stencils first check how many extrema are in a
 * given stencil. If the number of extrema is less than or equal to a
 * non-negative integer `max_number_of_extrema` which is given as an input
 * parameter, perform fifth order weighted compact nonlinear reconstruction with
 * Z oscillation indicator (WCNS-5Z); otherwise, switch to the monotonised
 * central (MC) reconstruction.
 *
 * See the documentations of `wcns5z()` and `monotonised_central()` for
 * descriptions on each reconstruction schemes.
 *
 */
template <size_t NonlinearWeightExponent, size_t Dim>
void wcns5z_mc(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const size_t max_number_of_extrema, const double epsilon) {
  detail::reconstruct<detail::Wcns5zMcReconstructor<NonlinearWeightExponent>>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables, max_number_of_extrema, epsilon);
}

/*!
 * \brief Returns function pointers to the `wcns5z_mc` function, lower neighbor
 * reconstruction, and upper neighbor reconstruction.
 *
 * This is useful for controlling template parameters like the
 * `NonlinearWeightExponent` from an input file by setting a function pointer.
 * Note that the reason the reconstruction functions instead of say the
 * `pointwise` member function is returned is to avoid function pointers inside
 * tight loops.
 */
template <size_t Dim>
auto wcns5z_mc_function_pointers(size_t nonlinear_weight_exponent)
    -> std::tuple<
        void (*)(gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                 gsl::not_null<std::array<gsl::span<double>, Dim>*>,
                 const gsl::span<const double>&,
                 const DirectionMap<Dim, gsl::span<const double>>&,
                 const Index<Dim>&, size_t, size_t, double),
        void (*)(gsl::not_null<DataVector*>, const DataVector&,
                 const DataVector&, const Index<Dim>&, const Index<Dim>&,
                 const Direction<Dim>&, const size_t&, const double&),
        void (*)(gsl::not_null<DataVector*>, const DataVector&,
                 const DataVector&, const Index<Dim>&, const Index<Dim>&,
                 const Direction<Dim>&, const size_t&, const double&)>;

}  // namespace fd::reconstruction
