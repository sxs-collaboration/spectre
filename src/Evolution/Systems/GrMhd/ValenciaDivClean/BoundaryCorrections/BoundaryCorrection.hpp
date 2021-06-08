// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/// \cond
class Hll;
class Rusanov;
/// \endcond

/*!
 * \brief The base class used to make boundary corrections factory createable so
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
  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Hll, Rusanov>;

  virtual std::unique_ptr<BoundaryCorrection> get_clone() const noexcept = 0;
};
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
