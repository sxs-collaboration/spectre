// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {

namespace Tags {
/// \brief Tag containing the quadrupole moment.
///
/// The quadrupole moment is defined as \f$\tilde{D} x^i x^j\f$
/// (equation 21 of \cite Shibata2003), with
/// \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, and
/// \f$x\f$ the coordinates in Frame Fr.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct QuadrupoleMoment : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Fr>;
};

/// \brief Tag containing the first time derivative of the quadrupole moment.
///
/// For the first derivative of the quadrupole moment, we use
/// \f$\tilde{D} (v^i x^j + x^i v^j)\f$ (equation 23 of \cite Shibata2003),
/// with \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, \f$x\f$ the
/// coordinates in Frame Fr, and \f$v\f$ the corresponding spatial velocity.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct QuadrupoleMomentDerivative : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Fr>;
};
}  // namespace Tags

/// \brief Function computing the quadrupole moment.
///
/// Computes the quadrupole moment, using
/// \f$\tilde{D} x^i x^j\f$ (equation 21 of \cite Shibata2003),
/// with \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, and
/// \f$x\f$ the coordinates in Frame Fr. Result of calculation stored in result.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
void quadrupole_moment(
    const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Fr>& coordinates);

/// \brief Function computing the first time derivative of the
/// quadrupole moment.
///
/// Computes the first time derivative of the quadrupole moment, using
/// \f$\tilde{D} (v^i x^j + x^i v^j)\f$ (equation 23 of \cite Shibata2003),
/// with \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, \f$x\f$ the
/// coordinates in Frame Fr, and \f$v\f$ the corresponding spatial velocity.
/// Result of calculation stored in result.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
void quadrupole_moment_derivative(
    const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Fr>& coordinates,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity);

} // namespace hydro
