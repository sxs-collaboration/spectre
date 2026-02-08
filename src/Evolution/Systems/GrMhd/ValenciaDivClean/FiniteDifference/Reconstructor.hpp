// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
/// \cond
class MonotonicityPreserving5Prim;
class MonotonisedCentralPrim;
class PositivityPreservingAdaptiveOrderPrim;
class Wcns5zPrim;
/// \endcond

/*!
 * \brief The base class from which all reconstruction schemes must inherit
 */
class Reconstructor : public SPECTRE_CHARM_PUPable(Reconstructor) {
 public:
  Reconstructor() = default;
  Reconstructor(const Reconstructor&) = default;
  Reconstructor& operator=(const Reconstructor&) = default;
  Reconstructor(Reconstructor&&) = default;
  Reconstructor& operator=(Reconstructor&&) = default;
  ~Reconstructor() override = default;

  SPECTRE_FINDUS_VIRTUAL()
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) SPECTRE_CHARM_OVERRIDE();

#if defined(SPECTRE_USE_CHARM)
  /// \cond
  WRAPPED_PUPable_abstract(Reconstructor);  // NOLINT
  /// \endcond
#endif  // SPECTRE_USE_CHARM

  using creatable_classes =
      tmpl::list<MonotonicityPreserving5Prim, MonotonisedCentralPrim,
                 PositivityPreservingAdaptiveOrderPrim, Wcns5zPrim>;

  virtual std::unique_ptr<Reconstructor> get_clone() const = 0;

  virtual size_t ghost_zone_size() const = 0;

  virtual bool supports_adaptive_order() const { return false; }

  virtual bool reconstruct_rho_times_temperature() const = 0;
};
}  // namespace grmhd::ValenciaDivClean::fd
