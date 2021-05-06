// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>  // for std::pair

#include "Domain/Structure/Side.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Index;
/// \endcond

namespace fd {
/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Variable and flux vector splitting reconstruction schemes for finite
 * difference methods.
 *
 * Implementations of reconstruction methods must call the function
 * `fd::reconstruction::detail::reconstruct` to perform the reconstruction.
 * This function performs the actual loops taking into account strides and ghost
 * zones for the reconstruction method. The `reconstruct` function has the
 * following signature:
 *
 * \code
 * template <typename Reconstructor, typename... ArgsForReconstructor, size_t
 *           Dim>
 * void reconstruct(
 *     const gsl::not_null<std::array<gsl::span<double>, Dim>*>
 *         reconstructed_upper_side_of_face_vars,
 *     const gsl::not_null<std::array<gsl::span<double>, Dim>*>
 *         reconstructed_lower_side_of_face_vars,
 *     const gsl::span<const double>& volume_vars,
 *     const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
 *     const Index<Dim>& volume_extents, const size_t number_of_variables,
 *     const ArgsForReconstructor&... args_for_reconstructor) noexcept;
 * \endcode
 *
 * The type of reconstruction done is specified with the `Reconstructor`
 * explicit template parameter. Parameters for the reconstruction scheme, such
 * as the linear weights and the epsilon tolerance in a CWENO reconstruction
 * scheme,  are forwarded along using the `args_for_reconstruction` parameter
 * pack.
 *
 * `Reconstructor` classes must define:
 * - a `static constexpr size_t stencil_width()` function that
 *   returns the size of the stencil, e.g. 3 for minmod, and 5 for 5-point
 *   reconstruction.
 * - a
 *   \code
 *      SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
 *            const double* const u, const int stride)
 *   \endcode
 *   function that optionally takes the additional arguments.
 *   The `u` are the cell-centered values to reconstruct. The value \f$u_i\f$ in
 *   the current FD cell is located at `u[0]`, the value at neighbor cell
 *   \f$u_{i+1}\f$ is at `u[stride]`, and the value at neighbor cell
 *   \f$u_{i-1}\f$ is at `u[-stride]`. The returned values are the
 *   reconstructed solution on the lower and upper side of the cell.
 *
 * \note Currently the stride is always one because we transpose the data before
 * reconstruction. However, it may be faster to have a non-unit stride without
 * the transpose. We have the `stride` parameter in the reconstruction schemes
 * to make testing performance easier in the future.
 *
 * Here is an ASCII illustration of the names of various quantities and where in
 * the cells they are:
 *
 * \code
 *   reconstructed_upper_side_of_face_vars v
 * reconstructed_lower_side_of_face_vars v
 *   volume_vars (at the cell-center) v
 *                               |    x   |
 *                                ^ reconstructed_upper_side_of_face_vars
 *                              ^ reconstructed_lower_side_of_face_vars
 * \endcode
 *
 * Notice that the `reconstructed_upper_side_of_face_vars` are on the lower side
 * of the cell, while the `reconstructed_lower_side_of_face_vars` are on the
 * upper side of the cell.
 */
namespace reconstruction {
namespace detail {
template <typename Reconstructor, size_t Dim, typename... ArgsForReconstructor>
void reconstruct(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const ArgsForReconstructor&... args_for_reconstructor) noexcept;
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief In a given direction, reconstruct the cells in the neighboring Element
 * (or cluster of cells) nearest to the shared boundary between the current and
 * neighboring Element (or cluster of cells).
 *
 * This is needed if one is sending reconstruction and flux data separately, or
 * if one is using DG-FD hybrid schemes. Below is an ASCII diagram of what is
 * reconstructed.
 *
 * ```
 *  Self      |  Neighbor
 *  x x x x x | o o o
 *            ^+
 *            Reconstruct to right/+ side of the interface
 * ```
 */
template <Side LowerOrUpperSide, typename Reconstructor, size_t Dim,
          typename... ArgsForReconstructor>
void reconstruct_neighbor(
    gsl::not_null<DataVector*> face_data, const DataVector& volume_data,
    const DataVector& neighbor_data, const Index<Dim>& volume_extents,
    const Index<Dim>& ghost_data_extents,
    const Direction<Dim>& direction_to_reconstruct,
    const ArgsForReconstructor&... args_for_reconstructor) noexcept;
}  // namespace reconstruction
}  // namespace fd
