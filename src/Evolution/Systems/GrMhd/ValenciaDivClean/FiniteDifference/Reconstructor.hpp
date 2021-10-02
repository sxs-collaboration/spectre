// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
/// \cond
class MonotisedCentralPrim;
/// \endcond

/*!
 * \brief The base class from which all reconstruction schemes must inherit
 */
class Reconstructor : public PUP::able {
 public:
  Reconstructor() = default;
  Reconstructor(const Reconstructor&) = default;
  Reconstructor& operator=(const Reconstructor&) = default;
  Reconstructor(Reconstructor&&) = default;
  Reconstructor& operator=(Reconstructor&&) = default;
  ~Reconstructor() override = default;

  /// \cond
  explicit Reconstructor(CkMigrateMessage* msg);
  WRAPPED_PUPable_abstract(Reconstructor);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<MonotisedCentralPrim>;

  virtual std::unique_ptr<Reconstructor> get_clone() const = 0;

  virtual size_t ghost_zone_size() const = 0;

  void pup(PUP::er& p) override;
};
}  // namespace grmhd::ValenciaDivClean::fd
