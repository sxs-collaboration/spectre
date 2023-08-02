// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"

/// \cond
class DataVector;
template <typename Frame>
class Strahlkorper;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace StrahlkorperGr {
/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Spin function of a 2D `Strahlkorper`.
 *
 * \details See Eqs. (2) and (10)
 * of \cite Owen2017yaj. This function computes the
 * "spin function," which is an ingredient for horizon surface integrals that
 * measure quasilocal spin. This function is proportional to the imaginary part
 * of the horizon's complex scalar curvature. For Kerr black holes, the spin
 * function is proportional to the horizon vorticity. It is also useful for
 * visualizing the direction of a black hole's spin.
 * Specifically, this function computes
 * \f$\Omega = \epsilon^{AB}\nabla_A\omega_B\f$,
 * where capital indices index the tangent bundle of the surface and
 * where \f$\omega_\mu=(K_{\rho\nu}-K g_{\rho\nu})h_\mu^\rho s^\nu\f$ is
 * the curl of the angular momentum density of the surface,
 * \f$h^\rho_\mu = \delta_\mu^\rho + u_\mu u^\rho - s_\mu s^\rho\f$
 * is the projector tangent to the 2-surface,
 * \f$g_{\rho\nu} = \psi_{\rho\nu} + u_\rho u_\nu\f$ is the spatial
 * metric of the spatial slice, \f$u^\rho\f$ is the unit normal to the
 * spatial slice, and \f$s^\nu\f$ is the unit normal vector to the surface.
 * Because the tangent basis vectors \f$e_A^\mu\f$ are
 * orthogonal to both \f$u^\mu\f$ and \f$s^\mu\f$, it is straightforward
 * to show that \f$\Omega = \epsilon^{AB} \nabla_A K_{B\mu}s^\mu\f$.
 * This function uses the tangent vectors of the `Strahlkorper` to
 * compute \f$K_{B\mu}s^\mu\f$ and then numerically computes the
 * components of its gradient. The argument `area_element`
 * can be computed via `StrahlkorperGr::area_element`.
 * The argument `unit_normal_vector` can be found by raising the
 * index of the one-form returned by `StrahlkorperGr::unit_normal_oneform`.
 * The argument `tangents` is a Tangents that can be obtained from the
 * StrahlkorperDataBox using the `StrahlkorperTags::Tangents` tag.
 */
template <typename Frame>
void spin_function(gsl::not_null<Scalar<DataVector>*> result,
                   const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
                   const Strahlkorper<Frame>& strahlkorper,
                   const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
                   const Scalar<DataVector>& area_element,
                   const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);

template <typename Frame>
Scalar<DataVector> spin_function(
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const Strahlkorper<Frame>& strahlkorper,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);
/// @}

/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Spin magnitude measured on a 2D `Strahlkorper`.
 *
 * \details Measures the quasilocal spin magnitude of a Strahlkorper, by
 * inserting \f$\alpha=1\f$ into Eq. (10) of \cite Owen2009sb
 * and dividing by \f$8\pi\f$ to yield the spin magnitude. The
 * spin magnitude is a Euclidean norm of surface integrals over the horizon
 * \f$S = \frac{1}{8\pi}\oint z \Omega dA\f$,
 * where \f$\Omega\f$ is obtained via `StrahlkorperGr::spin_function()`,
 * \f$dA\f$ is the area element, and \f$z\f$ (the "spin potential") is a
 * solution of a generalized eigenproblem given by Eq. (9) of
 * \cite Owen2009sb. Specifically,
 * \f$\nabla^4 z + \nabla \cdot (R\nabla z) = \lambda \nabla^2 z\f$, where
 * \f$R\f$ is obtained via `StrahlkorperGr::ricci_scalar()`. The spin
 * magnitude is the Euclidean norm of the three values of \f$S\f$ obtained from
 * the eigenvectors \f$z\f$ with the 3 smallest-magnitude
 * eigenvalues \f$\lambda\f$. Note that this formulation of the eigenproblem
 * uses the "Owen" normalization, Eq. (A9) and surrounding discussion in
 * \cite Lovelace2008tw.
 * The eigenvectors are normalized  with the "Kerr normalization",
 * Eq. (A22) of \cite Lovelace2008tw.
 * The argument `spatial_metric` is the metric of the 3D spatial slice
 * evaluated on the `Strahlkorper`.
 * The argument `tangents` can be obtained from the StrahlkorperDataBox
 * using the `StrahlkorperTags::Tangents` tag, and the argument
 * `unit_normal_vector` can
 * be found by raising the index of the one-form returned by
 * `StrahlkorperGr::unit_normal_one_form`.
 * The argument `strahlkorper` is the surface on which the spin magnitude is
 * computed.
 * The argument `area_element`
 * can be computed via `StrahlkorperGr::area_element`.
 */
template <typename Frame>
void dimensionful_spin_magnitude(
    gsl::not_null<double*> result, const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& area_element);

template <typename Frame>
double dimensionful_spin_magnitude(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& area_element);
/// @}

/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Dimensionless spin magnitude of a `Strahlkorper`.
 *
 * \details
 * This function computes the dimensionless spin magnitude \f$\chi\f$
 * from the dimensionful spin magnitude \f$S\f$ and the christodoulou
 * mass \f$M\f$ of a black hole. Specifically, computes
 * \f$\chi = \frac{S}{M^2}\f$.
 */

void dimensionless_spin_magnitude(const gsl::not_null<double*> result,
                                  const double dimensionful_spin_magnitude,
                                  const double christodoulou_mass);

double dimensionless_spin_magnitude(const double dimensionful_spin_magnitude,
                                    const double christodoulou_mass);
/// @}

/*!
 * \ingroup SurfacesGroup
 * \brief Spin vector of a 2D `Strahlkorper`.
 *
 * \details Computes the spin vector of a `Strahlkorper` in a
 * `MeasurementFrame`, such as `Frame::Inertial`. The result is a
 * `std::array<double, 3>` containing the Cartesian components (in
 * `MeasurementFrame`) of the spin vector whose magnitude is
 * `spin_magnitude`. `spin_vector` will return the dimensionless spin
 * components if `spin_magnitude` is the dimensionless spin magnitude,
 * and it will return the dimensionful spin components if
 * `spin_magnitude` is the dimensionful spin magnitude. The spin
 * vector is given by
 * a surface integral over the horizon \f$\mathcal{H}\f$ [Eq. (25) of
 * \cite Owen2017yaj]:
 * \f$S^i = \frac{S}{N} \oint_\mathcal{H} dA \Omega (x^i - x^i_0 - x^i_R) \f$,
 * where \f$S\f$ is the spin magnitude,
 * \f$N\f$ is a normalization factor enforcing \f$\delta_{ij}S^iS^j = S\f$,
 * \f$dA\f$ is the area element (via `StrahlkorperGr::area_element`),
 * \f$\Omega\f$ is the "spin function" (via `StrahlkorperGr::spin_function`),
 * \f$x^i\f$ are the `MeasurementFrame` coordinates of points on
 * the `Strahlkorper`,
 * \f$x^i_0\f$ are the `MeasurementFrame` coordinates of the center
 * of the Strahlkorper,
 * \f$x^i_R = \frac{1}{8\pi}\oint_\mathcal{H} dA (x^i - x^i_0) R \f$,
 * and \f$R\f$ is the intrinsic Ricci scalar of the `Strahlkorper`
 * (via `StrahlkorperGr::ricci_scalar`).
 * Note that measuring positions on the horizon relative to
 * \f$x^i_0 + x^i_R\f$ instead of \f$x^i_0\f$ ensures that the mass dipole
 * moment vanishes.
 *
 * \param result The computed spin vector in `MeasurementFrame`.
 * \param spin_magnitude The spin magnitude.
 * \param area_element The area element on `strahlkorper`'s
 *        collocation points.
 * \param ricci_scalar The intrinsic ricci scalar on `strahlkorper`'s
 *        collocation points.
 * \param spin_function The spin function on `strahlkorper`'s
 *        collocation points.
 * \param strahlkorper The Strahlkorper in the `MetricDataFrame` frame.
 * \param measurement_frame_coords The Cartesian coordinates of `strahlkorper`'s
 * collocation points, mapped to `MeasurementFrame`.
 *
 * Note that `spin_vector` uses two frames: the Strahlkorper and all of the
 * metric quantities are in `MetricDataFrame` and are used for doing integrals,
 * but the `measurement_frame_coordinates` are in `MeasurementFrame` and are
 * used for making sure the result is in the appropriate frame.  The two frames
 * `MeasurementFrame` and `MetricDataFrame` may or may not be the same.
 * In principle, spin_vector could be written using only a single frame
 * (`MeasurementFrame`) but that would require that the metric quantities
 * are known on the collocation points of a Strahlkorper in `MeasurementFrame`,
 * which would involve more interpolation.
 */
template <typename MetricDataFrame, typename MeasurementFrame>
void spin_vector(
    const gsl::not_null<std::array<double, 3>*> result, double spin_magnitude,
    const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const Strahlkorper<MetricDataFrame>& strahlkorper,
    const tnsr::I<DataVector, 3, MeasurementFrame>& measurement_frame_coords);

template <typename MetricDataFrame, typename MeasurementFrame>
std::array<double, 3> spin_vector(
    double spin_magnitude, const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const Strahlkorper<MetricDataFrame>& strahlkorper,
    const tnsr::I<DataVector, 3, MeasurementFrame>& measurement_frame_coords);
}  // namespace StrahlkorperGr
