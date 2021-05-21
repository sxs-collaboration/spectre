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
// @{
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
// @}
}  // namespace Bjorhus
}  // namespace GeneralizedHarmonic::BoundaryConditions
