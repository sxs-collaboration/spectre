// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Distribution.hpp"

#include <ostream>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::CoordinateMaps {

std::ostream& operator<<(std::ostream& os, const Distribution distribution) {
  switch (distribution) {
    case Distribution::Linear:
      return os << "Linear";
    case Distribution::Equiangular:
      return os << "Equiangular";
    case Distribution::Logarithmic:
      return os << "Logarithmic";
    case Distribution::Inverse:
      return os << "Inverse";
    case Distribution::Projective:
      return os << "Projective";
    default:
      ERROR("Unknown domain::CoordinateMaps::Distribution type");
  }
}

bool operator==(const DistributionAndSingularityPosition& lhs,
                const DistributionAndSingularityPosition& rhs) {
  return lhs.distribution == rhs.distribution and
         lhs.singularity_position == rhs.singularity_position;
}

}  // namespace domain::CoordinateMaps

template <>
domain::CoordinateMaps::Distribution
Options::create_from_yaml<domain::CoordinateMaps::Distribution>::create<void>(
    const Options::Option& options) {
  const auto distribution = options.parse_as<std::string>();
  if (distribution == "Linear") {
    return domain::CoordinateMaps::Distribution::Linear;
  } else if (distribution == "Equiangular") {
    return domain::CoordinateMaps::Distribution::Equiangular;
  } else if (distribution == "Logarithmic") {
    return domain::CoordinateMaps::Distribution::Logarithmic;
  } else if (distribution == "Inverse") {
    return domain::CoordinateMaps::Distribution::Inverse;
  } else if (distribution == "Projective") {
    return domain::CoordinateMaps::Distribution::Projective;
  }
  PARSE_ERROR(options.context(),
              "Distribution must be 'Linear', 'Equiangular', 'Logarithmic', "
              "'Inverse', or 'Projective'.");
}

// The helper classes exist to create a nested level of options like this:
// Distribution:
//   Logarithmic:
//     SingularityPosition: 0.
namespace domain::CoordinateMaps::detail {

struct SingularityPositionImpl {
  static constexpr Options::String help = {
      "Position of coordinate singularity."};
  struct SingularityPosition {
    using type = double;
    static constexpr Options::String help = {
        "Position of coordinate singularity. "
        "Must be outside the domain. "
        "A singularity position close to the lower or upper bound of the "
        "interval leads to very small grid spacing near that end, and "
        "placing the singularity further away from the domain increases the "
        "grid spacing. See the documentation of "
        "'domain::CoordinateMap::Distribution' for details."};
  };
  using options = tmpl::list<SingularityPosition>;
  double value;
};

template <Distribution Dist>
struct DistAndSingularityPositionImpl {
  static_assert(Dist == Distribution::Logarithmic or
                    Dist == Distribution::Inverse,
                "Singularity position is only required for 'Logarithmic' and "
                "'Inverse' grid point distributions.");
  static constexpr Options::String help = {
      "The singularity position for the 'Logarithmic' or 'Inverse' "
      "distribution."};
  struct DistAndSingularityPos {
    static std::string name() {
      if constexpr (Dist == Distribution::Logarithmic) {
        return "Logarithmic";

      } else {
        return "Inverse";
      }
    }
    using type = SingularityPositionImpl;
    static constexpr Options::String help{
        "The singularity position for the 'Logarithmic' or 'Inverse' "
        "distribution."};
  };
  using options = tmpl::list<DistAndSingularityPos>;
  SingularityPositionImpl singularity_position;
};

}  // namespace domain::CoordinateMaps::detail

template <>
domain::CoordinateMaps::DistributionAndSingularityPosition
Options::create_from_yaml<
    domain::CoordinateMaps::DistributionAndSingularityPosition>::
    create<void>(const Options::Option& options) {
  const auto dist = options.parse_as<std::variant<
      domain::CoordinateMaps::Distribution,
      domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
          domain::CoordinateMaps::Distribution::Logarithmic>,
      domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
          domain::CoordinateMaps::Distribution::Inverse>>>();
  if (std::holds_alternative<domain::CoordinateMaps::Distribution>(dist)) {
    const auto distribution =
        std::get<domain::CoordinateMaps::Distribution>(dist);
    if (distribution == domain::CoordinateMaps::Distribution::Linear or
        distribution == domain::CoordinateMaps::Distribution::Projective or
        distribution == domain::CoordinateMaps::Distribution::Equiangular) {
      return {distribution, std::nullopt};
    } else {
      PARSE_ERROR(
          options.context(),
          "The distribution '"
              << distribution
              << "' requires a singularity position. Specify it like this:\n  "
              << distribution << ":\n    SingularityPosition: 0.0");
    }
  } else if (std::holds_alternative<
                 domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
                     domain::CoordinateMaps::Distribution::Logarithmic>>(
                 dist)) {
    const auto& dist_and_singularity_position =
        std::get<domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
            domain::CoordinateMaps::Distribution::Logarithmic>>(dist);
    return {domain::CoordinateMaps::Distribution::Logarithmic,
            dist_and_singularity_position.singularity_position.value};
  } else if (std::holds_alternative<
                 domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
                     domain::CoordinateMaps::Distribution::Inverse>>(dist)) {
    const auto& dist_and_singularity_position =
        std::get<domain::CoordinateMaps::detail::DistAndSingularityPositionImpl<
            domain::CoordinateMaps::Distribution::Inverse>>(dist);
    return {domain::CoordinateMaps::Distribution::Inverse,
            dist_and_singularity_position.singularity_position.value};
  } else {
    PARSE_ERROR(options.context(),
                "Failed to parse distribution. Specify either a distribution "
                "such as 'Linear', or a distribution with its singularity "
                "position such as 'Logarithmic: SingularityPosition: 0.0'.");
  }
}
