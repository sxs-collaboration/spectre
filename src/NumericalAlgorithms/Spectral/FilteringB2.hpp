// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
class Matrix;
template <size_t>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace Spectral::filtering {
/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 basis functions, with an optional exponential roll-off and an
 * optional top-mode cutoff.
 *
 * \details Representing functions on a filled disk requires special basis
 * functions, namely ZernikeB2. These are inherently two-dimensional, meaning
 * the radial and angular spectral spaces are intertwined. This function goes
 * to that combined modal space, applies the filter, and transforms back.
 *
 * When `half_power` has a value, a smooth exponential roll-off (see
 * exponential_filter()) is applied to the combined radial-angular modes using
 * the coefficient `alpha`. When `num_modes_to_kill` is nonzero, the highest
 * `num_modes_to_kill` angular (Fourier \f$m\f$) modes are additionally set to
 * zero, with the \f$m=0\f$ mode always retained. If `half_power` is
 * `std::nullopt` and `num_modes_to_kill` is zero this function is a no-op.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_disk_filter(gsl::not_null<Variables<TagsList>*> u,
                            const Mesh<2>& mesh, double alpha,
                            std::optional<unsigned> half_power,
                            size_t num_modes_to_kill);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 basis functions, with an optional exponential roll-off and an
 * optional top-mode cutoff.
 *
 * \details Overload taking a caller-managed working buffer. Avoids heap
 * allocation when the filter is applied repeatedly (e.g. in
 * `Filters::FilledCylinder`).
 */
template <typename TagsList>
void zernike_b2_disk_filter(gsl::not_null<Variables<TagsList>*> u,
                            gsl::not_null<DataVector*> buf, const Mesh<2>& mesh,
                            double alpha, std::optional<unsigned> half_power,
                            size_t num_modes_to_kill);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 basis functions.
 *
 * \details Equivalent to `zernike_b2_disk_filter()` with `half_power` set and
 * no top-mode cutoff. Kept for callers that only want the exponential filter.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_disk_exponential_filter(gsl::not_null<Variables<TagsList>*> u,
                                        const Mesh<2>& mesh, double alpha,
                                        unsigned half_power);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 \f$\times\f$ Legendre basis functions, with optional independent
 * roll-offs for the combined radial-angular disk modes and the axial \f$z\f$
 * modes plus an optional angular top-mode cutoff.
 *
 * \details Representing functions on a filled cylinder requires special basis
 * functions, namely a filled disk with ZernikeB2 cross Legendre. This requires
 * inherently two-dimensional basis functions, meaning the radial and angular
 * spectral spaces are intertwined. This function goes to that combined modal
 * space, applies the disk filter, transforms back, and then filters the third
 * I1 dimension.
 *
 * The combined radial-angular (disk) modes are filtered with an exponential
 * roll-off when `radial_angular_half_power` has a value and with a top-mode
 * cutoff of the highest `num_modes_to_kill` angular (Fourier \f$m\f$) modes
 * when `num_modes_to_kill` is nonzero (the \f$m=0\f$ mode is always retained).
 * The axial \f$z\f$ direction is filtered with an independent exponential
 * roll-off when `z_half_power` has a value. The coefficient `alpha` is shared
 * by all exponential roll-offs. Any direction whose half-power is
 * `std::nullopt` (and, for the disk, with `num_modes_to_kill` zero) is left
 * untouched.
 *
 * While the radial-angular plane is fixed to ZernikeB2 \f$\times\f$ Fourier,
 * the axial \f$z\f$ filter is applied with exponential_filter() in whatever 1D
 * spectral basis the mesh uses in the \f$z\f$ direction, so both Legendre and
 * Chebyshev (with any quadrature) are supported there.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_cylinder_filter(
    gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh, double alpha,
    std::optional<unsigned> radial_angular_half_power,
    std::optional<unsigned> z_half_power, size_t num_modes_to_kill);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 \f$\times\f$ Legendre basis functions, with optional independent
 * roll-offs for the combined radial-angular disk modes and the axial \f$z\f$
 * modes plus an optional angular top-mode cutoff.
 *
 * \details Overload taking a caller-managed working buffer. Avoids heap
 * allocation when the filter is applied repeatedly (e.g. in
 * `Filters::FilledCylinder`). One can optionally pass the exponential filter
 * to apply in the z direction, otherwise it will be computed.
 *
 */
template <typename TagsList>
void zernike_b2_cylinder_filter(
    gsl::not_null<Variables<TagsList>*> u, gsl::not_null<DataVector*> buf,
    const Mesh<3>& mesh, double alpha,
    std::optional<unsigned> radial_angular_half_power,
    std::optional<unsigned> z_half_power, size_t num_modes_to_kill,
    const std::optional<Matrix>& z_filter);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 \f$\times\f$ Legendre basis functions.
 *
 * \details Equivalent to `zernike_b2_cylinder_filter()` with both the disk and
 * axial half-powers set to `half_power` and no top-mode cutoff. Kept for
 * callers that only want a single exponential filter applied everywhere.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_cylinder_exponential_filter(
    gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh, double alpha,
    unsigned half_power);
}  // namespace Spectral::filtering
