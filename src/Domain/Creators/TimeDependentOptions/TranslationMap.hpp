// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class to be used as an option for initializing translation map
 * coefficients.
 */
template <size_t Dim>
struct TranslationMapOptions {
  using type = Options::Auto<TranslationMapOptions, Options::AutoLabel::None>;
  static std::string name() { return "TranslationMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent translation of the coordinates. Specify "
      "'None' to not use this map."};

  struct InitialValues {
    using type = std::variant<std::array<std::array<double, Dim>, 3>,
                              FromVolumeFile<names::Translation>>;
    static constexpr Options::String help = {
        "Initial values for the translation map, its velocity and "
        "acceleration."};
  };

  using options = tmpl::list<InitialValues>;

  TranslationMapOptions() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  TranslationMapOptions(std::variant<std::array<std::array<double, Dim>, 3>,
                                     FromVolumeFile<names::Translation>>
                            values_from_options);

  std::array<DataVector, 3> initial_values{};
};
}  // namespace domain::creators::time_dependent_options
