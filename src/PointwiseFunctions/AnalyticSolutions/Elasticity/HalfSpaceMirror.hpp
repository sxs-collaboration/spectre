// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
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
 * \brief The solution for a deformed half-space mirror.
 *
 * \details This solution describes the displacement of a semi-infinite mirror
 * with a perpendicularly acting force at its center. The beam material is
 * characterized by an isotropic homogeneous constitutive relation
 * \f$Y^{ijkl}\f$ (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`). In this scenario,
 * the force introduces a pressure in the form of a Gaussian distribution
 *
 * \f{align}
 * T^{zz} &= \frac{e^{-\frac{\omega^2}{\omega_0^2}}}{\pi\omega_0^2} F \\
 * T^{xy} &= T^{yy} = 0 \text{.}
 * \f}
 *
 * as a Neumann boundary condition. The implementation here matches with
 * Eqs. 11.93 and 11.97 in \cite ThorneBlandford2017.
 */

class HalfSpaceMirror {
 public:
  static constexpr size_t dim = 3;

  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<dim>;

  struct BeamWidth {
    using type = double;
    static constexpr OptionString help{"The lasers beam width"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct AppliedForce {
    using type = double;
    static constexpr OptionString help{"The applied force"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct Material {
    using type = constitutive_relation_type;
    static constexpr OptionString help{"The material properties of the beam"};
  };

  struct IntegrationIntervals {
    using type = size_t;
    static constexpr OptionString help{"Intervals used in Integration"};
    static type lower_bound() noexcept { return 1; }
  };

  struct IntergrationTolerance {
    using type = double;
    static constexpr OptionString help{"Tolerance used in Integration"};
    static type lower_bound() noexcept { return 0.; }
    static type upper_bound() noexcept { return 1e-5; }
  };

  using options = tmpl::list<BeamWidth, AppliedForce, Material,
                             IntegrationIntervals, IntergrationTolerance>;
  static constexpr OptionString help{
      "A semi-infinite mirror on which a laser introduces stress perpendicular "
      "to the mirrors surface. The displacement then is the Hankel-Transform "
      "of the general solution multiplied by the beam profile"};

  HalfSpaceMirror() = default;
  HalfSpaceMirror(const HalfSpaceMirror&) noexcept = delete;
  HalfSpaceMirror& operator=(const HalfSpaceMirror&) noexcept = delete;
  HalfSpaceMirror(HalfSpaceMirror&&) noexcept = default;
  HalfSpaceMirror& operator=(HalfSpaceMirror&&) noexcept = default;
  ~HalfSpaceMirror() noexcept = default;

  HalfSpaceMirror(double beam_width, double applied_force,
                  constitutive_relation_type constitutive_relation,
                  size_t no_intervals, double absolute_tolerance) noexcept;

  HalfSpaceMirror(double beam_width, double applied_force, double bulk_modulus,
                  double shear_modulus, size_t no_intervals,
                  double absolute_tolerance) noexcept;

  const constitutive_relation_type& constitutive_relation() const noexcept {
    return constitutive_relation_;
  }

  // @{
  /// Retrieve variable at coordinates `x`
  auto variables(const tnsr::I<DataVector, dim>& x,
                 tmpl::list<Tags::Displacement<dim>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Displacement<dim>>;

  auto variables(const tnsr::I<DataVector, dim>& x,
                 tmpl::list<Tags::Strain<dim>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Strain<dim>>;

  auto variables(const tnsr::I<DataVector, dim>& x,
                 tmpl::list<Tags::Stress<dim>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Stress<dim>>;

  auto variables(
      const tnsr::I<DataVector, dim>& x,
      tmpl::list<::Tags::FixedSource<Tags::Displacement<dim>>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<dim>>>;
  // @}

  /// Initial guess for variables
  auto variables(
      const tnsr::I<DataVector, dim>& x,
      tmpl::list<::Tags::Initial<Tags::Displacement<dim>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::Initial<Tags::Displacement<dim>>>;

  auto variables(const tnsr::I<DataVector, dim>& x,
                 tmpl::list<::Tags::Initial<Tags::Strain<dim>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::Initial<Tags::Strain<dim>>>;

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1, "An unsupported Tag was requested.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const HalfSpaceMirror& lhs,
                         const HalfSpaceMirror& rhs) noexcept;

  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
  double applied_force_{std::numeric_limits<double>::signaling_NaN()};
  constitutive_relation_type constitutive_relation_{};
  size_t no_intervals_{};
  double absolute_tolerance_{};
};

bool operator!=(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept;

}  // namespace Solutions
}  // namespace Elasticity
