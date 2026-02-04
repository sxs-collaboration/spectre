// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace evolution {
/*!
 * \brief The base class used to create boundary corrections from input files
 * and store them in the global cache.
 */
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;
  ~BoundaryCorrection() override = default;

  /// \cond
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace evolution
