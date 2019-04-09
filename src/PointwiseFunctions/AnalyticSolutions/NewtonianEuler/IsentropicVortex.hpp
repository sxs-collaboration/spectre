// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare NewtonianEuler::Solutions::IsentropicVortex::IntermediateVariables

namespace NewtonianEuler {
namespace Solutions {

/*!
 * \brief Newtonian isentropic vortex in Cartesian coordinates
 *
 * The analytic solution to the 2-D Newtonian Euler system
 * representing the slow advection of an incompressible, isentropic
 * vortex \cite Yee1999. The initial condition is the superposition of a
 * mean uniform flow with a gaussian-profile vortex. When embedded in
 * 3-D space, the isentropic vortex is still a solution to the corresponding 3-D
 * system if the velocity along the third axis is a constant. In Cartesian
 * coordinates \f$(x, y, z)\f$, and using dimensionless units, the primitive
 * quantities at a given time \f$t\f$ are then
 *
 * \f{align*}
 * \rho &= \left[1 - \dfrac{(\gamma - 1)\beta^2}{8\gamma\pi^2}\exp\left(
 * 1 - r^2\right)\right]^{1/(\gamma - 1)}, \\
 * v_x &= U - \dfrac{\beta\tilde y}{2\pi}\exp\left(\dfrac{1 - r^2}{2}\right),\\
 * v_y &= V + \dfrac{\beta\tilde x}{2\pi}\exp\left(\dfrac{1 - r^2}{2}\right),\\
 * v_z &= W,\\
 * \epsilon &= \frac{\rho^{\gamma - 1}}{\gamma - 1},
 * \f}
 *
 * with
 *
 * \f{align*}
 * r^2 &= {\tilde x}^2 + {\tilde y}^2,\\
 * \tilde x &= x - X_0 - U t,\\
 * \tilde y &= y - Y_0 - V t,
 * \f}
 *
 * where \f$(X_0, Y_0)\f$ is the position of the vortex on the \f$(x, y)\f$
 * plane at \f$t = 0\f$, \f$(U, V, W)\f$ are the components of the mean flow
 * velocity, \f$\beta\f$ is the vortex strength, and \f$\gamma\f$ is the
 * adiabatic index. The pressure \f$p\f$ is then obtained from the dimensionless
 * polytropic relation
 *
 * \f{align*}
 * p = \rho^\gamma.
 * \f}
 *
 * On the other hand, if the velocity along the \f$z-\f$axis is not a constant
 * but a function of the \f$z\f$ coordinate, the resulting modified isentropic
 * vortex is still a solution to the Newtonian Euler system, but with source
 * terms that are proportional to \f$dv_z/dz\f$. (See
 * NewtonianEuler::Sources::IsentropicVortexSource.) For testing purposes,
 * we choose to write the velocity as a uniform field plus a periodic
 * perturbation,
 *
 * \f{align*}
 * v_z(z) = W + \epsilon \sin{z},
 * \f}
 *
 * where \f$\epsilon\f$ is the amplitude of the perturbation. The resulting
 * source for the Newtonian Euler system will then be proportional to
 * \f$\epsilon \cos{z}\f$.
 */
template <size_t Dim>
class IsentropicVortex {
  template <typename DataType>
  struct IntermediateVariables;

 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<false>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"The adiabatic index of the fluid."};
  };

  /// The position of the center of the vortex at \f$t = 0\f$
  struct Center {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {
        "The coordinates of the center of the vortex at t = 0."};
  };

  /// The mean flow velocity.
  struct MeanVelocity {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {"The mean flow velocity."};
  };

  /// The amplitude of the perturbation generating a source term.
  struct PerturbAmplitude {
    using type = double;
    static constexpr OptionString help = {
        "The amplitude of the perturbation producing sources."};
  };

  /// The strength of the vortex.
  struct Strength {
    using type = double;
    static constexpr OptionString help = {"The strength of the vortex."};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<AdiabaticIndex, Center, MeanVelocity,
                             PerturbAmplitude, Strength>;
  static constexpr OptionString help = {
      "Newtonian Isentropic Vortex. Works in 2 and 3 dimensions."};

  IsentropicVortex() = default;
  IsentropicVortex(const IsentropicVortex& /*rhs*/) = delete;
  IsentropicVortex& operator=(const IsentropicVortex& /*rhs*/) = delete;
  IsentropicVortex(IsentropicVortex&& /*rhs*/) noexcept = default;
  IsentropicVortex& operator=(IsentropicVortex&& /*rhs*/) noexcept = default;
  ~IsentropicVortex() = default;

  IsentropicVortex(double adiabatic_index,
                   const std::array<double, Dim>& center,
                   const std::array<double, Dim>& mean_velocity,
                   double perturbation_amplitude, double strength);

  /// Retrieve a collection of hydrodynamic variables at position x and time t
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      const double t,  // NOLINT
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    IntermediateVariables<DataType> vars(x, t, center_, mean_velocity_,
                                         perturbation_amplitude_, strength_);
    return {tuples::get<Tags>(variables(tmpl::list<Tags>{}, vars))...};
  }

  const EquationsOfState::PolytropicFluid<false>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/,
      const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(tmpl::list<Tags::Pressure<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;
  // @}

  // Intermediate variables needed to compute the primitives
  template <typename DataType>
  struct IntermediateVariables {
    IntermediateVariables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                          double t, const std::array<double, Dim>& center,
                          const std::array<double, Dim>& mean_velocity,
                          double perturbation_amplitude,
                          double strength) noexcept;
    DataType x_tilde{};
    DataType y_tilde{};
    DataType profile{};
    // (3D only) Extra term in the velocity along z that generates sources.
    DataType perturbation{};
  };

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const IsentropicVortex<SpatialDim>& lhs,
      const IsentropicVortex<SpatialDim>& rhs) noexcept;

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, Dim> center_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, Dim> mean_velocity_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double strength_ = std::numeric_limits<double>::signaling_NaN();

  // This is an ideal gas undergoing an isentropic process,
  // so the relation between the pressure and the mass density is polytropic,
  // where the polytropic exponent corresponds to the adiabatic index.
  EquationsOfState::PolytropicFluid<false> equation_of_state_{};
};

template <size_t Dim>
bool operator!=(const IsentropicVortex<Dim>& lhs,
                const IsentropicVortex<Dim>& rhs) noexcept;

}  // namespace Solutions
}  // namespace NewtonianEuler
