// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// Contains all variable fixers.
namespace VariableFixing {
namespace OptionTags {
/// \ingroup VariableFixingGroup
/// \brief The radius at which to begin applying the lower bound.
///
/// \see RadiallyFallingFloor to see where this is used.
struct MaskRadius {
  static constexpr OptionString help =
      "The radius at which to begin applying the lower bound";
  using type = double;
};
}  // namespace OptionTags

/// \ingroup VariableFixingGroup
/// \brief Applies a pressure and density floor dependent on the distance
/// to the origin.
///
/// Applies the floors:
/// \f$\rho(r) \geq \rho_{\mathrm{fl}}(r) = 10^{-5}r^{-3/2}\f$
/// and \f$P(r) \geq P_{\mathrm{fl}}(r) = \frac{1}{3} \times 10^{-7}r^{-5/2}\f$
/// when \f$ r > \f$`radius_at_which_to_begin_applying_floor`.
/// These bounds are described in Porth et al.'s ["The Black Hole Accretion
/// Code"](https://arxiv.org/pdf/1611.09720.pdf).
template <size_t Dim, typename Density, typename Pressure>
struct RadiallyFallingFloor {
  using return_tags = tmpl::list<Density, Pressure>;
  using argument_tags = tmpl::list<::Tags::Coordinates<Dim, Frame::Inertial>>;
  using const_global_cache_tag_list = tmpl::list<OptionTags::MaskRadius>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> density,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const double radius_at_which_to_begin_applying_floor) noexcept {
    const auto radii = magnitude(coords);
    for (size_t i = 0; i < density->get().size(); i++) {
      if (UNLIKELY(radii.get()[i] < radius_at_which_to_begin_applying_floor)) {
        continue;
      }
      const double& radius = radii.get()[i];
      const double radius_to_the_three_halves_power = sqrt(radius) * radius;
      pressure->get()[i] =
          std::max(pressure->get()[i],
                   (1.e-7 / 3.) / (radius * radius_to_the_three_halves_power));
      density->get()[i] =
          std::max(density->get()[i], 1.e-5 / radius_to_the_three_halves_power);
    }
  }
};
}  // namespace VariableFixing
