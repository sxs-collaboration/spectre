// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

namespace gh::gauges {
/*!
 * \brief Compute \f$0.5 n^a n^b \Pi_{ab}\f$ and \f$0.5 n^a n^b \Phi_{iab}\f$
 */
template <size_t Dim, typename Frame>
void half_pi_and_phi_two_normals(
    gsl::not_null<Scalar<DataVector>*> half_pi_two_normals,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame>*> half_phi_two_normals,
    const tnsr::A<DataVector, Dim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataVector, Dim, Frame>& pi,
    const tnsr::iaa<DataVector, Dim, Frame>& phi);
}  // namespace gh::gauges
