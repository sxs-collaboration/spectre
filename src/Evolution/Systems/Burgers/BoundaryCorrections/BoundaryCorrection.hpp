// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace Burgers::BoundaryCorrections {
/// \cond
class Hll;
class Rusanov;
/// \endcond

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

  explicit BoundaryCorrection(CkMigrateMessage* msg) : PUP::able(msg) {}

  /// \cond
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Hll, Rusanov>;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace Burgers::BoundaryCorrections
