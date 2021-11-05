// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <pup_stl.h>

#include "Parallel/CharmPupable.hpp"

namespace evolution {
/*!
 * \brief Namespace for things related to initial data used for evolution
 * systems.
 */
namespace initial_data {
/*!
 * \brief The abstract base class for initial data of evolution systems. All
 * analytic solutions and analytic data must virtually inherit from this class.
 */
class InitialData : public PUP::able {
 protected:
  InitialData() = default;

 public:
  ~InitialData() override = default;

  /// \cond
  explicit InitialData(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(InitialData);
  /// \endcond
};
}  // namespace initial_data
}  // namespace evolution
