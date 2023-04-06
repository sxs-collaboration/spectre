// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
/*!
 * \brief Boundary corrections/numerical fluxes for the GRFFE sytem.
 */
namespace BoundaryCorrections {

/// \cond
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

  /// \cond
  explicit BoundaryCorrection(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Rusanov>;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const = 0;
};
}  // namespace BoundaryCorrections
}  // namespace ForceFree
