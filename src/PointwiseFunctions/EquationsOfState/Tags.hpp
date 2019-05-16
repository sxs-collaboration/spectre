// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// Can be used to retrieve the equation of state from the cache without having
/// to know the template parameters of EquationOfState.
struct EquationOfStateBase {};

/// \ingroup OptionTagsGroup
/// The equation of state, with `IsRelativistic` and `ThermodynamicDim` left
/// as template parameters.
template <bool IsRelativistic, size_t ThermodynamicDim>
struct EquationOfState : EquationOfStateBase {
  static constexpr OptionString help = "The equation of state.";
  using type = std::unique_ptr<
      EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>>;
};
}  // namespace OptionTags
