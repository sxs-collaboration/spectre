// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "Domain/Structure/Direction.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Mesh;

namespace boost {
template <class T>
struct hash;
}  // namespace boost
/// \endcond

namespace Limiters::Weno_detail {

// Compute the WENO weighted reconstruction of a DataVector, see e.g.,
// Eq. 4.3 of Zhu2016. This is fairly standard, though different references can
// differ in their choice of oscillation/smoothness indicator. The
// `DerivativeWeight` enum specifies the relative weight of each derivative
// term when computing the oscillation indicator.
//
// The reconstruction modifies the DataVector in `local_polynomial` by adding
// to it a weighted combination of one or more neighbor contributions, passed
// in as several DataVectors in `neighbor_polynommials`. Each neighbor
// polynomial must have the same mean as the local polynomial; this is checked
// with an ASSERT.
template <size_t VolumeDim>
void reconstruct_from_weighted_sum(
    gsl::not_null<DataVector*> local_polynomial, double neighbor_linear_weight,
    DerivativeWeight derivative_weight, const Mesh<VolumeDim>& mesh,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_polynomials) noexcept;

}  // namespace Limiters::Weno_detail
