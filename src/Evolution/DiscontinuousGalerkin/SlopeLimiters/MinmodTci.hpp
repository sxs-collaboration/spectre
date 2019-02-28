// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <utility>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class Mesh;
/// \endcond

namespace SlopeLimiters {
namespace Minmod_detail {

// Implements the troubled-cell indicator corresponding to the Minmod limiter.
//
// The troubled-cell indicator (TCI) determines whether or not limiting is
// needed. See SlopeLimiters::Minmod for a full description of the Minmod
// limiter. Note that as an optimization, this TCI returns (by reference) some
// additional data that are used by the Minmod limiter in the case where the
// TCI returns true (i.e., the case where limiting is needed).
template <size_t VolumeDim>
bool troubled_cell_indicator(
    gsl::not_null<double*> u_mean,
    gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    gsl::not_null<DataVector*> u_lin_buffer,
    gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const DataVector& u, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept;

}  // namespace Minmod_detail
}  // namespace SlopeLimiters
