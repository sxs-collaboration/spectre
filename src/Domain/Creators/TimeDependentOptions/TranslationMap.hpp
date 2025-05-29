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
 * \brief Class that holds hard coded translation map options from the input
 * file.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 */
template <size_t Dim>
struct TranslationMapOptions {
  using type =
      Options::Auto<std::variant<TranslationMapOptions<Dim>, FromVolumeFile>,
                    Options::AutoLabel::None>;
  static std::string name() { return "TranslationMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent translation of the coordinates. Specify "
      "'None' to not use this map."};

  struct InitialValues {
    using type = std::array<std::array<double, Dim>, 3>;
    static constexpr Options::String help = {
        "Initial values for the translation map. You can optionally specify "
        "its first two time derivatives. If time derivatives aren't specified, "
        "zero will be used."};
  };

  using options = tmpl::list<InitialValues>;

  TranslationMapOptions() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  TranslationMapOptions(
      const std::array<std::array<double, Dim>, 3>& initial_values_in,
      const Options::Context& context = {});

  std::array<DataVector, 3> initial_values{};
};

/*!
 * \brief Helper function that takes the variant of the translation map options,
 * and returns the fully constructed translation function of time.
 *
 * \details The function of time will have a new \p initial_time and \p
 * expiration_time, unless it is read from a file and is replaying.
 */
template <size_t Dim>
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_translation(
    const std::variant<TranslationMapOptions<Dim>, FromVolumeFile>&
        translation_map_options,
    double initial_time, double expiration_time);
}  // namespace domain::creators::time_dependent_options
