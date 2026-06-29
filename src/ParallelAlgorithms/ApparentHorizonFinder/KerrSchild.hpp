// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "NumericalAlgorithms/Strahlkorper/InitialShape.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::InitialShapes {
/*!
 * \ingroup SurfacesGroup
 * \brief An initial shape for a Strahlkorper corersponding to a Kerr-Schild
 * black hole.
 */
template <typename Frame>
class KerrSchild : public ylm::InitialShape<Frame> {
 public:
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the Strahlkorper expansion"};
  };
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
  };
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"Dimensionless spin vector"};
  };

  using options = tmpl::list<Center, Mass, Spin>;
  static constexpr Options::String help = {
      "Construct a Kerr-Schild horizon for the given mass and dimensionless "
      "spin vector."};
  static std::string name() { return "KerrSchild"; }

  KerrSchild() = default;
  KerrSchild(std::array<double, 3> center, double mass,
             std::array<double, 3> spin);

  /// \cond
  explicit KerrSchild(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KerrSchild);  // NOLINT
  /// \endcond

  ylm::Strahlkorper<Frame> strahlkorper(
      size_t l_max, const Options::Context& context) const override;

  void pup(PUP::er& p) override;

 private:
  std::array<double, 3> center_{};
  double mass_{};
  std::array<double, 3> spin_{};
};
}  // namespace ah::InitialShapes
