// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
/// \cond
template <typename System>
class MonotonisedCentralPrim;
template <typename System>
class PositivityPreservingAdaptiveOrderPrim;
template <typename System>
class Wcns5zPrim;
/// \endcond

/*!
 * \brief The base class from which all reconstruction schemes must inherit
 */

// template on System instead
template <typename System>
class Reconstructor : public SPECTRE_CHARM_PUPable(Reconstructor<System>) {
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

  using system = System;
  using creatable_classes =
      tmpl::list<MonotonisedCentralPrim<System>,
                 PositivityPreservingAdaptiveOrderPrim<System>,
                 Wcns5zPrim<System>>;

  virtual std::unique_ptr<Reconstructor<System>> get_clone() const = 0;

  virtual size_t ghost_zone_size() const = 0;

  virtual bool supports_adaptive_order() const { return false; }

  virtual bool reconstruct_rho_times_temperature() const = 0;
};
}  // namespace grmhd::GhValenciaDivClean::fd
