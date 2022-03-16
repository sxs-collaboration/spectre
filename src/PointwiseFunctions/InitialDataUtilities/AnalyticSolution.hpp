// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"

namespace elliptic::analytic_data {
/*!
 * \brief Subclasses represent analytic solutions of elliptic systems.
 *
 * Subclasses must define the compile-time interfaces of both
 * `elliptic::analytic_data::InitialGuess` and
 * `elliptic::analytic_data::Background`.
 *
 * The combined set of system variables and background fields must solve the
 * elliptic PDEs. Subclasses must list all additional requirements needed so
 * they solve the elliptic PDEs in the class documentation and in their option
 * help-string.
 */
class AnalyticSolution : public elliptic::analytic_data::InitialGuess,
                         public elliptic::analytic_data::Background {
 protected:
  AnalyticSolution() = default;

 public:
  ~AnalyticSolution() override = default;

  /// \cond
  explicit AnalyticSolution(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(AnalyticSolution);
  /// \endcond

  virtual std::unique_ptr<AnalyticSolution> get_clone() const = 0;
};
}  // namespace elliptic::analytic_data
