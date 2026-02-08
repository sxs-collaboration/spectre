// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Options/String.hpp"

namespace Ccz4::fd {
/*!
 * \brief Dummy reconstructor just to return the ghost zone size.
 *
 * This class is needed to use the SetInterpolators action in
 * dg-subcell infrastructure.
 */
class DummyReconstructor
    : public SPECTRE_CHARM_DERIVED(DummyReconstructor, Reconstructor) {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help{
      "Dummy reconstructor that allows using the subcell infrastructure."};

  DummyReconstructor() = default;
  DummyReconstructor(DummyReconstructor&&) = default;
  DummyReconstructor& operator=(DummyReconstructor&&) = default;
  DummyReconstructor(const DummyReconstructor&) = default;
  DummyReconstructor& operator=(const DummyReconstructor&) = default;
  ~DummyReconstructor() override = default;

  WRAPPED_PUPable_decl_base_template(Reconstructor, DummyReconstructor);

  auto get_clone() const -> std::unique_ptr<Reconstructor> override;

  void pup(PUP::er& p) override;

  // 3 ghost pts needed for 6th order KO filter
  size_t ghost_zone_size() const override { return 3; }
};

}  // namespace Ccz4::fd
