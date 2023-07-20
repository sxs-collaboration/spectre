// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::fd {
/// \cond
class MonotonisedCentral;
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

  void pup(PUP::er& p) override;

  /// \cond
  explicit Reconstructor(CkMigrateMessage* msg);
  WRAPPED_PUPable_abstract(Reconstructor);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<MonotonisedCentral>;

  virtual std::unique_ptr<Reconstructor> get_clone() const = 0;

  virtual size_t ghost_zone_size() const = 0;

  virtual bool supports_adaptive_order() const { return false; }
};
}  // namespace ForceFree::fd
