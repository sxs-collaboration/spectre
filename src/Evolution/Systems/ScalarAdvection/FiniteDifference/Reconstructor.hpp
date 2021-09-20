// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::fd {
/*!
 * \brief The base class from which all reconstruction schemes must inherit
 */
template <size_t Dim>
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
  explicit Reconstructor(CkMigrateMessage* msg) noexcept;
  WRAPPED_PUPable_abstract(Reconstructor<Dim>);  // NOLINT
  /// \endcond

  using creatable_classes = tmpl::list<>;

  virtual std::unique_ptr<Reconstructor<Dim>> get_clone() const noexcept = 0;

  virtual size_t ghost_zone_size() const noexcept = 0;
};
}  // namespace ScalarAdvection::fd
