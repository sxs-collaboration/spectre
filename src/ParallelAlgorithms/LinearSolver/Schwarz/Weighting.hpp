// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Utilities/Gsl.hpp"

namespace LinearSolver::Schwarz {

/*!
 * \brief Weights for the solution on an element-centered subdomain, decreasing
 * from 1 to 0.5 towards the `side` over the logical distance `width`, and
 * further to 0 over the same distance outside the element.
 *
 * The weighting function over a full element-centered subdomain is
 *
 * \f{equation}
 * w(\xi) = \frac{1}{2}\left( \phi\left( \frac{\xi + 1}{\delta} \right) -
 * \phi\left( \frac{\xi - 1}{\delta} \right) \right) \f}
 *
 * where \f$\phi(\xi)\f$ is a second-order `::smoothstep`, i.e. the quintic
 * polynomial
 *
 * \f{align*}
 * \phi(\xi) = \begin{cases} \mathrm{sign}(\xi) \quad \text{for}
 * \quad |\xi| > 1 \\
 * \frac{1}{8}\left(15\xi - 10\xi^3 + 3\xi^5\right) \end{cases}
 * \f}
 *
 * (see Eq. (39) in \cite Vincent2019qpd).
 *
 * The `LinearSolver::Schwarz::extruding_weight` and
 * `LinearSolver::Schwarz::intruding_weight` functions each compute one of the
 * two terms in \f$w(\xi)\f$. For example, consider an element-centered
 * subdomain `A` that overlaps with a neighboring element-centered subdomain
 * `B`. To combine solutions on `A` and `B` to a weighted solution on `A`,
 * multiply the solution on `A` with the `extruding_weight` and the solution on
 * `B` with the `intruding_weight`, both evaluated at the logical coordinates in
 * `A` and at the `side` of `A` that faces `B`.
 */
DataVector extruding_weight(const DataVector& logical_coords, double width,
                            const Side& side) noexcept;

/// @{
/*!
 * \brief Weights for data on the central element of an element-centered
 * subdomain
 *
 * Constructs the weighting field
 *
 * \f{equation}
 * W(\boldsymbol{\xi}) = \prod^d_{i=0} w(\xi^i)
 * \f}
 *
 * where \f$w(\xi^i)\f$ is the one-dimensional weighting function described in
 * `LinearSolver::Schwarz::extruding_weight` and \f$\xi^i\f$ are the
 * element-logical coordinates (see Eq. (41) in \cite Vincent2019qpd).
 */
template <size_t Dim>
void element_weight(
    gsl::not_null<Scalar<DataVector>*> element_weight,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const std::array<double, Dim>& overlap_widths,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept;

template <size_t Dim>
Scalar<DataVector> element_weight(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const std::array<double, Dim>& overlap_widths,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept;
/// @}

/*!
 * \brief Weights for the intruding solution of a neighboring element-centered
 * subdomain, increasing from 0 to 0.5 towards the `side` over the logical
 * distance `width`, and further to 1 over the same distance outside the
 * element.
 *
 * \see `LinearSolver::Schwarz::extruding_weight`
 */
DataVector intruding_weight(const DataVector& logical_coords, double width,
                            const Side& side) noexcept;

/// @{
/*!
 * \brief Weights for data on overlap regions intruding into an element-centered
 * subdomain
 *
 * Constructs the weighting field \f$W(\xi)\f$ as described in
 * `LinearSolver::Schwarz::element_weight` for the data that overlaps with the
 * central element of an element-centered subdomain. The weights are constructed
 * in such a way that all weights at a grid point sum to one, i.e. the weight is
 * conserved. The `logical_coords` are the element-logical coordinates of the
 * central element.
 *
 * This function assumes that corner- and edge-neighbors of the central element
 * are not part of the subdomain, which means that no contributions from those
 * neighbors are expected although the weighting field is non-zero in overlap
 * regions with those neighbors. Therefore, to retain conservation we must
 * account for this missing weight by adding it to the central element, to the
 * intruding overlaps from face-neighbors, or split it between the two. We
 * choose to add the weight to the intruding overlaps, since that's where
 * information from the corner- and edge-regions propagates through in a DG
 * context.
 */
template <size_t Dim>
void intruding_weight(
    gsl::not_null<Scalar<DataVector>*> weight,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const Direction<Dim>& direction,
    const std::array<double, Dim>& overlap_widths,
    size_t num_intruding_overlaps,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept;

template <size_t Dim>
Scalar<DataVector> intruding_weight(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const Direction<Dim>& direction,
    const std::array<double, Dim>& overlap_widths,
    size_t num_intruding_overlaps,
    const std::unordered_set<Direction<Dim>>& external_boundaries) noexcept;
/// @}

}  // namespace LinearSolver::Schwarz
