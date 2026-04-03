// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::fd {
/// \cond
template <size_t Dim>
class AoWeno53Prim;
template <size_t Dim>
class MonotonisedCentralPrim;
/// \endcond

/*!
 * \brief The base class from which all reconstruction schemes must inherit
 *
 * Currently we have hard-coded reconstructing \f$\rho, p, v^i\f$. However, the
 * DG-subcell solver is coded generally enough that an efficient implementation
 * of reconstructing the conserved or characteristic variables is also possible.
 * It is not yet clear how much info about what is being reconstructed is needed
 * at compile time and so we currently append `Prim` to the end of the
 * reconstruction schemes to clarify that they are reconstructing the primitive
 * variables. Ideally the choice of what variables to reconstruct can be made by
 * a runtime argument to the individual reconstruction schemes.
 */
template <size_t Dim>
class Reconstructor : public SPECTRE_CHARM_PUPable(Reconstructor<Dim>) {
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
      tmpl::list<AoWeno53Prim<Dim>, MonotonisedCentralPrim<Dim>>;

  virtual std::unique_ptr<Reconstructor<Dim>> get_clone() const = 0;

  virtual size_t ghost_zone_size() const = 0;
};
}  // namespace NewtonianEuler::fd
