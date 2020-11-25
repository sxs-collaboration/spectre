// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Hydro/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace RelativisticEuler::Solutions {

/*!
 * \brief Smooth wave propagating in Minkowski spacetime.
 *
 * The relativistic Euler equations in Minkowski spacetime accept a
 * solution with constant pressure and uniform spatial velocity provided
 * that the rest mass density satisfies the advection equation
 *
 * \f{align*}{
 * \partial_t\rho + v^i\partial_i\rho = 0,
 * \f}
 *
 * and the specific internal energy is a function of the rest mass density only,
 * \f$\epsilon = \epsilon(\rho)\f$. For testing purposes, this class implements
 * this solution for the case where \f$\rho\f$ is a sine wave. The user
 * specifies the mean flow velocity of the fluid, the wavevector of the density
 * profile, and the amplitude \f$A\f$ of the density profile. In Cartesian
 * coordinates \f$(x, y, z)\f$, and using dimensionless units, the primitive
 * variables at a given time \f$t\f$ are then
 *
 * \f{align*}{
 * \rho(\vec{x},t) &= 1 + A \sin(\vec{k}\cdot(\vec{x} - \vec{v}t)) \\
 * \vec{v}(\vec{x},t) &= [v_x, v_y, v_z]^{T},\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho}\\
 * \f}
 *
 * where we have assumed \f$\epsilon\f$ and \f$\rho\f$ to be related through an
 * equation mathematically equivalent to the equation of state of an ideal gas,
 * where the pressure is held constant.
 */
template <size_t Dim>
class SmoothFlow : virtual public MarkAsAnalyticSolution,
                   private hydro::Solutions::SmoothFlow<Dim, true> {
  using smooth_flow = hydro::Solutions::SmoothFlow<Dim, true>;

 public:
  using options = typename smooth_flow::options;

  static constexpr Options::String help = {
      "Smooth flow in Minkowski spacetime."};

  SmoothFlow() = default;
  SmoothFlow(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow& operator=(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow(SmoothFlow&& /*rhs*/) noexcept = default;
  SmoothFlow& operator=(SmoothFlow&& /*rhs*/) noexcept = default;
  ~SmoothFlow() = default;

  SmoothFlow(const std::array<double, Dim>& mean_velocity,
             const std::array<double, Dim>& wavevector, double pressure,
             double adiabatic_index, double perturbation_size) noexcept;

  explicit SmoothFlow(CkMigrateMessage* msg) noexcept;

  using smooth_flow::equation_of_state;
  using typename smooth_flow::equation_of_state_type;

  // Overload the variables function from the base class.
  using smooth_flow::variables;

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim>& x, const double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, Dim>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const SmoothFlow<SpatialDim>& lhs,
      const SmoothFlow<SpatialDim>& rhs) noexcept;

  gr::Solutions::Minkowski<Dim> background_spacetime_{};
};

template <size_t Dim>
bool operator!=(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept;
}  // namespace RelativisticEuler::Solutions
