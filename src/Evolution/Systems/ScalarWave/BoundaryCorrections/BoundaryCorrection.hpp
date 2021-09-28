// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace ScalarWave::BoundaryCorrections {
/// \cond
template <size_t Dim>
class UpwindPenalty;
/// \endcond

/*!
 * \brief The base class used to make boundary corrections factory createable so
 * they can be specified in the input file.
 */
template <size_t Dim>
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
  WRAPPED_PUPable_abstract(BoundaryCorrection<Dim>);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<UpwindPenalty<Dim>>;

  virtual std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const = 0;
};
}  // namespace ScalarWave::BoundaryCorrections
