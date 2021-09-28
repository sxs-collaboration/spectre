// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace RadiationTransport::M1Grey::BoundaryCorrections {
/// \cond
template <typename NeutrinoSpeciesList>
class Rusanov;
/// \endcond

/*!
 * \brief The base class used to create boundary corrections from input files
 * and store them in the global cache.
 */
template <typename NeutrinoSpeciesList>
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
  WRAPPED_PUPable_abstract(BoundaryCorrection<NeutrinoSpeciesList>);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Rusanov<NeutrinoSpeciesList>>;

  virtual std::unique_ptr<BoundaryCorrection<NeutrinoSpeciesList>> get_clone()
      const = 0;
};
}  // namespace RadiationTransport::M1Grey::BoundaryCorrections
