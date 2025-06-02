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
 * \brief Class that holds hard coded skew map options from the input
 * file.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 */
struct SkewMapOptions {
 private:
  struct Y {};
  struct Z {};

 public:
  using type = Options::Auto<std::variant<SkewMapOptions, FromVolumeFile>,
                             Options::AutoLabel::None>;
  static std::string name() { return "SkewMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent skew of the x coordinate (y and z "
      "coordinates are left unchanged). Specify 'None' to not use this map."};

  template <typename Coord>
  struct InitialValues {
    using type = std::array<double, 3>;
    static std::string name() {
      return "InitialValues" + pretty_type::name<Coord>();
    }
    static constexpr Options::String help =
        "Initial value, and two time derivatives.";
  };

  using options = tmpl::list<InitialValues<Y>, InitialValues<Z>>;

  SkewMapOptions() = default;
  SkewMapOptions(const std::array<double, 3>& initial_angles_y_in,
                 const std::array<double, 3>& initial_angles_z_in);

  std::array<double, 3> initial_angles_y{};
  std::array<double, 3> initial_angles_z{};
};

/*!
 * \brief Helper function that takes the variant of the skew map options,
 * and returns the fully constructed skew function of time.
 *
 * \details The function of time will have a new \p initial_time and \p
 * expiration_time, unless it is read from a file and is replaying.
 */
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_skew(
    const std::variant<SkewMapOptions, FromVolumeFile>& skew_map_options,
    double initial_time, double expiration_time);
}  // namespace domain::creators::time_dependent_options
