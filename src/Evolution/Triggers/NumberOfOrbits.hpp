// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Tags {
struct Time;
}  // namespace Tags

namespace Triggers {
class NumberOfOrbits : public Trigger {
 public:
  /// \cond
  NumberOfOrbits() = default;
  explicit NumberOfOrbits(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumberOfOrbits);  // NOLINT
  /// \endcond

  struct Value {
    using type = double;
    static constexpr Options::String help = {
        "Number of orbits to compare against."};
  };

  using options = tmpl::list<Value>;
  static constexpr Options::String help{
      "Trigger when the evolution has reached a desired number of orbits."};

  explicit NumberOfOrbits(double number_of_orbits);

  using argument_tags = tmpl::list<Tags::Time, domain::Tags::FunctionsOfTime>;

  bool operator()(const double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const;

  // NOLINENEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double number_of_orbits_{};
};
}  // namespace Triggers
