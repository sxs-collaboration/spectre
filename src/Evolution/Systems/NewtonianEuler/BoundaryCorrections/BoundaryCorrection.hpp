// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Boundary corrections/numerical fluxes
namespace NewtonianEuler::BoundaryCorrections {
/// \cond
template <size_t Dim>
class Hll;
template <size_t Dim>
class Hllc;
template <size_t Dim>
class Rusanov;
/// \endcond

/*!
 * \brief The base class used to create boundary corrections from input files
 * and store them in the global cache.
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
  explicit BoundaryCorrection(CkMigrateMessage* msg) noexcept
      : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BoundaryCorrection<Dim>);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<Hll<Dim>, Hllc<Dim>, Rusanov<Dim>>;

  virtual std::unique_ptr<BoundaryCorrection<Dim>> get_clone()
      const noexcept = 0;
};
}  // namespace NewtonianEuler::BoundaryCorrections
