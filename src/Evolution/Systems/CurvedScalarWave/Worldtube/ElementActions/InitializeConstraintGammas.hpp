// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

/*!
 * \ingroup InitializationGroup
 * \brief Initializes the constraint damping parameters \f$\gamma_1\f$ and
 * \f$\gamma_2\f$.
 *
 * \details The constraint damping parameters  of the CurvedScalarWave system
 * are initialized according to a Gaussian
 * \f{align*}
 *  \gamma_1 &= 0 \\
 *  \gamma_2 &= A e^{-(\sigma r)^2 } + c,
 * \f}
 * where \f$r = \sqrt{\delta_{ij} x^i x^j}\f$ and with \f$A = 10\f$, \f$\sigma =
 * 10^{-1}\f$, \f$c = 10^{-4}\f$. These values were experimentally found to
 * ensure a stable evolution when using the worldtube scheme.
 *
 *  DataBox changes:
 * - Adds:
 *   * `CurvedScalarWave::Tags::ConstraintGamma1`
 *   * `CurvedScalarWave::Tags::ConstraintGamma2`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <size_t Dim>
struct InitializeConstraintDampingGammas {
  using return_tags = tmpl::list<CurvedScalarWave::Tags::ConstraintGamma1,
                                 CurvedScalarWave::Tags::ConstraintGamma2>;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using simple_tags = return_tags;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords);
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
