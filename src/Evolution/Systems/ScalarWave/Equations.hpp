// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
/*!
 * \brief A relic of an old incorrect way of handling boundaries for
 * non-conservative systems.
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
  using argument_tags = tmpl::list<Tags::Pi>;
  static void apply(gsl::not_null<Scalar<DataVector>*> psi_normal_dot_flux,
                    gsl::not_null<Scalar<DataVector>*> pi_normal_dot_flux,
                    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
                        phi_normal_dot_flux,
                    const Scalar<DataVector>& pi);
};
}  // namespace ScalarWave
