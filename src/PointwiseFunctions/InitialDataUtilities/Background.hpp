// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace elliptic::analytic_data {
/*!
 * \brief Subclasses supply variable-independent background data for an elliptic
 * solve.
 *
 * Examples for background fields are a background metric, associated curvature
 * quantities, matter sources such as a mass-density in the XCTS equations, or
 * just a source function \f$f(x)\f$ in a Poisson equation \f$\Delta u =
 * f(x)\f$.
 *
 * Subclasses must define the following compile-time interface:
 *
 * - They are option-creatable.
 * - They define a `variables` function that provides the fixed-sources in the
 *   elliptic equations (see `elliptic::protocols::FirstOrderSystem`). The
 *   function must have this signature:
 *
 *   \snippet Test_InitializeFixedSources.cpp background_vars_fct
 *
 *   It must support being called with a `tmpl::list` of all system-variable
 *   tags prefixed with `::Tags::FixedSource`.
 * - They define a `variables` function that provides data for all background
 *   quantities, if any are listed in the `background_fields` of the system (see
 *   `elliptic::protocols::FirstOrderSystem`). The function must have this
 *   signature:
 *
 *   \snippet Test_SubdomainOperator.cpp background_vars_fct_derivs
 *
 *   It must support being called with a `tmpl::list` of all background tags. It
 *   may use the `mesh` and `inv_jacobian` to compute numerical derivatives.
 */
class Background : public virtual PUP::able {
 protected:
  Background() = default;

 public:
  ~Background() override = default;

  /// \cond
  explicit Background(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(Background);
  /// \endcond
};
}  // namespace elliptic::analytic_data
