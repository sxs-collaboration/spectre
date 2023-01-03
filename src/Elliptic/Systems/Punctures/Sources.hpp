// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Punctures {

/*!
 * \brief Add the nonlinear sources for the puncture equation.
 *
 * Adds $\beta \left(\alpha \left(1 + u\right) + 1\right)^{-7}$.
 *
 * \see Punctures
 */
void add_sources(gsl::not_null<Scalar<DataVector>*> puncture_equation,
                 const Scalar<DataVector>& alpha,
                 const Scalar<DataVector>& beta,
                 const Scalar<DataVector>& field);

/*!
 * \brief Add the linearized sources for the puncture equation.
 *
 * Adds $\frac{d}{du}(\beta \left(\alpha \left(1 + u\right) + 1\right)^{-7})$.
 *
 * \see Punctures
 */
void add_linearized_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_puncture_equation,
    const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
    const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction);

/// The sources \f$S\f$ for the first-order formulation of the puncture equation
///
/// \see elliptic::protocols::FirstOrderSystem
struct Sources {
  using argument_tags = tmpl::list<Tags::Alpha, Tags::Beta>;
  static void apply(gsl::not_null<Scalar<DataVector>*> puncture_equation,
                    const Scalar<DataVector>& alpha,
                    const Scalar<DataVector>& beta,
                    const Scalar<DataVector>& field,
                    const tnsr::I<DataVector, 3>& field_flux);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, 3>*> equation_for_field_gradient,
      const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
      const Scalar<DataVector>& field);
};

/// The linearization of the sources \f$S\f$ for the first-order formulation of
/// the puncture equation
///
/// \see elliptic::protocols::FirstOrderSystem
struct LinearizedSources {
  using argument_tags = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Field>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> linearized_puncture_equation,
      const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
      const Scalar<DataVector>& field,
      const Scalar<DataVector>& field_correction,
      const tnsr::I<DataVector, 3>& field_flux_correction);
  static void apply(gsl::not_null<tnsr::i<DataVector, 3>*>
                        equation_for_field_gradient_correction,
                    const Scalar<DataVector>& alpha,
                    const Scalar<DataVector>& beta,
                    const Scalar<DataVector>& field,
                    const Scalar<DataVector>& field_correction);
};

}  // namespace Punctures
