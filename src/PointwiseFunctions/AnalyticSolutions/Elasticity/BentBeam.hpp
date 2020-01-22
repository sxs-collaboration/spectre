// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"   // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"      // IWYU pragma: keep
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity {
namespace Solutions {
/*!
 * \brief A state of pure bending of an elastic beam in 2D
 *
 * \details This solution describes a 2D slice through an elastic beam of length
 * \f$L\f$ and height \f$H\f$ that is subject to a bending moment \f$M=\int
 * T^{xx}y\mathrm{d}y\f$ (see e.g. \cite ThorneBlandford2017, Eq. 11.41c for a
 * bending moment in 1D). The beam material is characterized by an isotropic and
 * homogeneous constitutive relation \f$Y^{ijkl}\f$ in the plane-stress
 * approximation (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`). In this scenario,
 * no body-forces \f$f_\mathrm{ext}^j\f$ act on the material, so the
 * \ref Elasticity equations reduce to \f$\nabla_i T^{ij}=0\f$, but the bending
 * moment \f$M\f$ generates the stress
 *
 * \f{align}
 * T^{xx} &= \frac{12 M}{H^3} y \\
 * T^{xy} &= 0 = T^{yy} \text{.}
 * \f}
 *
 * By fixing the rigid-body motions to
 *
 * \f[
 * \xi^x(0,y)=0 \quad \text{and} \quad \xi^y\left(\pm \frac{L}{2},0\right)=0
 * \f]
 *
 * we find that this stress is produced by the displacement field
 *
 * \f{align}
 * \xi^x&=-\frac{12 M}{EH^3}xy \\
 * \xi^y&=\frac{6 M}{EH^3}\left(x^2+\nu y^2-\frac{L^2}{4}\right)
 * \f}
 *
 * in terms of the Young's modulus \f$E\f$ and the Poisson ration \f$\nu\f$ of
 * the material. The corresponding strain \f$S_{ij}=\partial_{(i}\xi_{j)}\f$ is
 *
 * \f{align}
 * S_{xx} &= -\frac{12 M}{EH^3} y \\
 * S_{yy} &= \frac{12 M}{EH^3} \nu y \\
 * S_{xy} &= S_{yx} = 0 \text{.}
 * \f}
 */
class BentBeam {
 public:
  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>;

  struct Length {
    using type = double;
    static constexpr OptionString help{"The beam length"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Height {
    using type = double;
    static constexpr OptionString help{"The beam height"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct BendingMoment {
    using type = double;
    static constexpr OptionString help{
        "The bending moment applied to the beam"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct Material {
    using type = constitutive_relation_type;
    static constexpr OptionString help{"The material properties of the beam"};
  };

  using options = tmpl::list<Length, Height, BendingMoment, Material>;
  static constexpr OptionString help{
      "A 2D slice through an elastic beam which is subject to a bending "
      "moment. The bending moment is applied along the length of the beam, "
      "i.e. the x-axis, so that the beam's left and right ends are bent "
      "towards the positive y-axis. It is measured in units of force."};

  BentBeam() = default;
  BentBeam(const BentBeam&) noexcept = delete;
  BentBeam& operator=(const BentBeam&) noexcept = delete;
  BentBeam(BentBeam&&) noexcept = default;
  BentBeam& operator=(BentBeam&&) noexcept = default;
  ~BentBeam() noexcept = default;

  BentBeam(double length, double height, double bending_moment,
           constitutive_relation_type constitutive_relation) noexcept;

  const constitutive_relation_type& constitutive_relation() const noexcept {
    return constitutive_relation_;
  }

  // @{
  /// Retrieve variable at coordinates `x`
  auto variables(const tnsr::I<DataVector, 2>& x,
                 tmpl::list<Tags::Displacement<2>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Displacement<2>>;

  auto variables(const tnsr::I<DataVector, 2>& x,
                 tmpl::list<Tags::Strain<2>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Strain<2>>;

  auto variables(const tnsr::I<DataVector, 2>& x,
                 tmpl::list<Tags::Stress<2>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Stress<2>>;

  auto variables(
      const tnsr::I<DataVector, 2>& x,
      tmpl::list<::Tags::FixedSource<Tags::Displacement<2>>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<2>>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 2>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const BentBeam& lhs, const BentBeam& rhs) noexcept;

  double length_{std::numeric_limits<double>::signaling_NaN()};
  double height_{std::numeric_limits<double>::signaling_NaN()};
  double bending_moment_{std::numeric_limits<double>::signaling_NaN()};
  constitutive_relation_type constitutive_relation_{};
};

bool operator!=(const BentBeam& lhs, const BentBeam& rhs) noexcept;

}  // namespace Solutions
}  // namespace Elasticity
