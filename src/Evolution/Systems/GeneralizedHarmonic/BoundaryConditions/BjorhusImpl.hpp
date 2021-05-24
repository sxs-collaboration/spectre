// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic::BoundaryConditions {
/// \brief Detailed implementation of Bjorhus-type boundary corrections
namespace Bjorhus {
/*!
 * \brief Computes the expression needed to set boundary conditions on the time
 * derivative of the characteristic field \f$v^{\psi}_{ab}\f$
 *
 * \details In the Bjorhus scheme, the time derivatives of evolved variables are
 * characteristic projected. A constraint-preserving correction term is added
 * here to the resulting characteristic (time-derivative) field:
 *
 * \f{align}
 * \Delta \partial_t v^{\psi}_{ab} = \lambda_{\psi} n^i C_{iab}
 * \f}
 *
 * where \f$n^i\f$ is the local unit normal to the external boundary,
 * \f$C_{iab} = \partial_i \psi_{ab} - \Phi_{iab}\f$ is the three-index
 * constraint, and \f$\lambda_{\psi}\f$ is the characteristic speed of the field
 * \f$v^{\psi}_{ab}\f$.
 */
template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_psi(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*> bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept;

/*!
 * \brief Computes the expression needed to set boundary conditions on the time
 * derivative of the characteristic field \f$v^{0}_{iab}\f$
 *
 * \details In the Bjorhus scheme, the time derivatives of evolved variables are
 * characteristic projected. A constraint-preserving correction term is added
 * here to the resulting characteristic (time-derivative) field \f$v^0_{iab}\f$:
 *
 * \f{align}
 * \Delta \partial_t v^{0}_{iab} = \lambda_{0} n^j C_{jiab}
 * \f}
 *
 * where \f$n^i\f$ is the local unit normal to the external boundary,
 * \f$C_{ijab} = \partial_i\Phi_{jab} - \partial_j\Phi_{iab}\f$ is the
 * four-index constraint, and \f$\lambda_{0}\f$ is the characteristic speed of
 * the field \f$v^0_{iab}\f$.
 *
 * \note In 3D, only the non-zero and unique components of the four-index
 * constraint are stored, as \f$\hat{C}_{iab} = \epsilon_i^{jk} C_{jkab}\f$. In
 * 2D the input expected here is \f$\hat{C}_{0ab} = C_{01ab}, \hat{C}_{1ab} =
 * C_{10ab}\f$, and in 1D \f$\hat{C}_{0ab} = 0\f$.
 */
template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_zero(
    gsl::not_null<tnsr::iaa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_zero,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept;
}  // namespace Bjorhus
}  // namespace GeneralizedHarmonic::BoundaryConditions
