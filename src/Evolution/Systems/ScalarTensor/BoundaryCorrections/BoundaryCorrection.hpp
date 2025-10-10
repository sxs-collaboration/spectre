// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// Boundary corrections/numerical fluxes
namespace ScalarTensor::BoundaryCorrections {
/*!
 * \brief The base class used to make boundary corrections factory creatable so
 * they can be specified in the input file.
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
  explicit BoundaryCorrection(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace ScalarTensor::BoundaryCorrections
