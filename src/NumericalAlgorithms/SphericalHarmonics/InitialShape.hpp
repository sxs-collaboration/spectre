// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ylm {
template <typename Frame>
class Strahlkorper;

/*!
 * \ingroup SurfacesGroup
 * \brief Base class for constructing initial Strahlkorper shapes.
 * \details When constructing a Stahlkorper from options (such as the
 * initial guess for an apparent horizon), the user can specify a resolution
 * and a shape (such as spherical, or read from file).
 */
template <typename Frame>
class InitialShape : public PUP::able {
 protected:
  /// \cond
  InitialShape() = default;
  InitialShape(const InitialShape&) = default;
  InitialShape(InitialShape&&) = default;
  InitialShape& operator=(const InitialShape&) = default;
  InitialShape& operator=(InitialShape&&) = default;
  /// \endcond

 public:
  ~InitialShape() override = default;
  explicit InitialShape(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(InitialShape);

  /// Construct a Strahlkorper with both `l_max` and `m_max` set to `l_max`.
  /// The `context` is used to report option-parsing errors.
  virtual Strahlkorper<Frame> strahlkorper(
      size_t l_max, const Options::Context& context) const = 0;
};

namespace InitialShapes {
/*!
 * \ingroup SurfacesGroup
 * \brief A spherical initial Strahlkorper shape.
 */
template <typename Frame>
class Sphere : public InitialShape<Frame> {
 public:
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center of the Strahlkorper expansion"};
  };
  struct Radius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of spherical Strahlkorper"};
  };

  using options = tmpl::list<Center, Radius>;
  static constexpr Options::String help = {
      "Construct a spherical Strahlkorper."};
  static std::string name() { return "Sphere"; }

  Sphere() = default;
  Sphere(std::array<double, 3> center, double radius);

  /// \cond
  explicit Sphere(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Sphere);  // NOLINT
  /// \endcond

  Strahlkorper<Frame> strahlkorper(
      size_t l_max, const Options::Context& context) const override;

  void pup(PUP::er& p) override;

 private:
  std::array<double, 3> center_{};
  double radius_{};
};

}  // namespace InitialShapes
}  // namespace ylm
