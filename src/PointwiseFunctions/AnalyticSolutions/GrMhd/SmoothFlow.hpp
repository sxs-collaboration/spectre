// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd {
namespace Solutions {

/*!
 * \brief Periodic GrMhd solution in Minkowski spacetime.
 *
 * An analytic solution to the 3-D GrMhd system. The user specifies the mean
 * flow velocity of the fluid, the wavevector of the density profile, and the
 * amplitude \f$A\f$ of the density profile. The magnetic field is taken to be
 * zero everywhere. In Cartesian coordinates \f$(x, y, z)\f$, and using
 * dimensionless units, the primitive quantities at a given time \f$t\f$ are
 * then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= 1 + A \sin(\vec{k}\cdot(\vec{x} - \vec{v}t)) \\
 * \vec{v}(\vec{x},t) &= [v_x, v_y, v_z]^{T},\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho}\\
 * \vec{B}(\vec{x},t) &= [0, 0, 0]^{T}
 * \f}
 */
class SmoothFlow : virtual public MarkAsAnalyticSolution,
                   public RelativisticEuler::Solutions::SmoothFlow<3> {
  using smooth_flow = RelativisticEuler::Solutions::SmoothFlow<3>;

 public:
  using options = smooth_flow::options;

  static constexpr Options::String help = {
      "Periodic smooth flow in Minkowski spacetime with zero magnetic field."};

  SmoothFlow() = default;
  SmoothFlow(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow& operator=(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow(SmoothFlow&& /*rhs*/) = default;
  SmoothFlow& operator=(SmoothFlow&& /*rhs*/) = default;
  ~SmoothFlow() = default;

  SmoothFlow(const std::array<double, 3>& mean_velocity,
             const std::array<double, 3>& wavevector, double pressure,
             double adiabatic_index, double perturbation_size);

  using smooth_flow::equation_of_state;
  using smooth_flow::equation_of_state_type;

  // Overload the variables function from the base class.
  using smooth_flow::variables;

  /// @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;
  /// @}

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/);  //  NOLINT

 protected:
  friend bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs);
};

bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs);
}  // namespace Solutions
}  // namespace grmhd
