// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

/// Items related to pointwise analytic data for elliptic solves, such as
/// initial guesses, analytic solutions, and background quantities in elliptic
/// PDEs
namespace elliptic::analytic_data {
/*!
 * \brief Subclasses represent an initial guess for an elliptic solve.
 *
 * Subclasses must define the following compile-time interface:
 *
 * - They are option-creatable.
 * - They define a `variables` function that can provide data for all variables
 *   that are being solved for. Specifically the function must have this
 *   signature:
 *
 *   \snippet Test_InitializeFields.cpp initial_guess_vars_fct
 *
 *   It must support being called with a `tmpl::list` of all tags that are being
 *   solved for. For this purpose it can be convenient to template the function
 *   on the set of requested tags.
 */
class InitialGuess : public virtual PUP::able {
 protected:
  InitialGuess() = default;

 public:
  ~InitialGuess() override = default;

  /// \cond
  explicit InitialGuess(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(InitialGuess);
  /// \endcond
};
}  // namespace elliptic::analytic_data
