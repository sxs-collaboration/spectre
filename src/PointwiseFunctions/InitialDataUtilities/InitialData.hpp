// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <pup_stl.h>

#include "Parallel/CharmPupable.hpp"

namespace InitialDataUtilities {
/*!
 * \ingroup InitialDataGroup
 * \brief The abstract base class for initial data of evolution systems.
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
}  // namespace InitialDataUtilities
