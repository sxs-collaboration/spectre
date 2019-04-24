// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

// We want the code to be nicely formatted for the documentation, not here.
// clang-format off
namespace {
double working_without_databoxes_1() noexcept {
/// [working_without_databoxes_small_program_1]
// Set up variables:
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;

// Use variables:
const double mass = density * volume;
const double acceleration = velocity * velocity / radius;
return mass * acceleration;
/// [working_without_databoxes_small_program_1]
}

/// [working_without_databoxes_mass_compute]
double mass_compute(const double density, const double volume) noexcept {
  return density * volume;
}
/// [working_without_databoxes_mass_compute]

/// [working_without_databoxes_accel_compute]
double acceleration_compute(
    const double velocity, const double radius) noexcept {
  return velocity * velocity / radius;
}
/// [working_without_databoxes_accel_compute]

double working_without_databoxes_2() noexcept {
/// [working_without_databoxes_small_program_2]
// Set up variables:
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;

// Use variables:
const double mass = mass_compute(density, volume);
const double acceleration = acceleration_compute(velocity, radius);
return mass * acceleration;
/// [working_without_databoxes_small_program_2]
}

/// [working_without_databoxes_force_compute]
double force_compute(const double velocity, const double radius,
                     const double density, const double volume) noexcept {
  const double mass = mass_compute(density, volume);
  const double acceleration = acceleration_compute(velocity, radius);
  return mass *  acceleration;
}
/// [working_without_databoxes_force_compute]

void failed_acceleration() noexcept {
/// [working_without_databoxes_failed_accel]
const double velocity = 4.0;
const double radius = 2.0;
const double acceleration = acceleration_compute(velocity, radius);
const double failed_acceleration = acceleration_compute(radius, velocity);
/// [working_without_databoxes_failed_accel]
  CHECK(not(acceleration == failed_acceleration));
}

double std_map_databox_1() noexcept {
/// [std_map_databox_small_program_1]
// Set up variables:
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;

// Set up databox:
std::map<std::string, double> naive_databox;
naive_databox["Velocity"] = velocity;
naive_databox["Radius"] = radius;
naive_databox["Density"] = density;
naive_databox["Volume"] = volume;
/// [std_map_databox_small_program_1]
  return naive_databox["Density"] * naive_databox["Volume"] *
         naive_databox["Velocity"] * naive_databox["Velocity"] /
         naive_databox["Radius"];
}

/// [std_map_databox_mass_compute]
double mass_compute(const std::map<std::string, double>& box) noexcept {
  return box.at("Density") * box.at("Volume");
}
/// [std_map_databox_mass_compute]

/// [std_map_databox_accel_compute]
double acceleration_compute(const std::map<std::string, double>& box) noexcept {
  return box.at("Velocity") * box.at("Velocity") / box.at("Radius");
}
/// [std_map_databox_accel_compute]

/// [std_map_databox_force_compute]
double force_compute(const std::map<std::string, double>& box) noexcept {
  const double mass = mass_compute(box);
  const double acceleration = acceleration_compute(box);
  return mass * acceleration;
}
/// [std_map_databox_force_compute]

double std_map_databox_2() noexcept {
/// [std_map_databox_small_program_2]
// Set up variables:
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;

// Set up databox:
std::map<std::string, double> naive_databox;
naive_databox["Velocity"] = velocity;
naive_databox["Radius"] = radius;
naive_databox["Density"] = density;
naive_databox["Volume"] = volume;

// Use variables:
return force_compute(naive_databox);
/// [std_map_databox_small_program_2]
}

bool std_tuple_databox_example() noexcept {
/// [std_tuple_databox_1]
std::tuple<double, size_t, bool> sophomore_databox =
  std::make_tuple(1.2, 8, true);
/// [std_tuple_databox_1]

/// [std_tuple_databox_2]
const bool bool_quantity = std::get<bool>(sophomore_databox);
// value obtained is `true`
/// [std_tuple_databox_2]
return bool_quantity;
}

namespace sophomore {
/// [std_tuple_tags]
struct Velocity{};
struct Radius{};
struct Density{};
struct Volume{};
/// [std_tuple_tags]

double std_tuple_databox_1() noexcept {
/// [std_tuple_small_program_1]
std::tuple<std::pair<Velocity,double>,
           std::pair<Radius, double>, std::pair<Density, double>,
           std::pair<Volume, double>> sophomore_databox =
  std::make_tuple(std::make_pair(Velocity{}, 4.0),
                  std::make_pair(Radius{}, 2.0), std::make_pair(Density{}, 0.5),
                  std::make_pair(Volume{}, 10.0));
/// [std_tuple_small_program_1]
  return std::get<std::pair<Density, double>>(sophomore_databox).second *
         std::get<std::pair<Volume, double>>(sophomore_databox).second *
         std::get<std::pair<Velocity, double>>(sophomore_databox).second *
         std::get<std::pair<Velocity, double>>(sophomore_databox).second /
         std::get<std::pair<Radius, double>>(sophomore_databox).second;
}

/// [std_tuple_mass_compute]
template<typename... Pairs>
double mass_compute(const std::tuple<Pairs...>& box) noexcept {
  return std::get<std::pair<Density, double>>(box).second *
         std::get<std::pair<Volume, double>>(box).second;
}
/// [std_tuple_mass_compute]

/// [std_tuple_acceleration_compute]
template<typename... Pairs>
double acceleration_compute(const std::tuple<Pairs...>& box) noexcept {
  return std::get<std::pair<Velocity, double>>(box).second *
         std::get<std::pair<Velocity, double>>(box).second /
         std::get<std::pair<Radius, double>>(box).second;
}
/// [std_tuple_acceleration_compute]

/// [std_tuple_force_compute]
template<typename... Pairs>
double force_compute(const std::tuple<Pairs...>& box) noexcept {
  const double mass = mass_compute(box);
  const double acceleration = acceleration_compute(box);
  return mass * acceleration;
}
/// [std_tuple_force_compute]

double std_tuple_databox_2() noexcept {
/// [std_tuple_small_program_2]
std::tuple<std::pair<Velocity,double>,
           std::pair<Radius, double>, std::pair<Density, double>,
           std::pair<Volume, double>> sophomore_databox =
  std::make_tuple(std::make_pair(Velocity{}, 4.0),
                  std::make_pair(Radius{}, 2.0), std::make_pair(Density{}, 0.5),
                  std::make_pair(Volume{}, 10.0));

/// [std_tuple_small_program_2]
  return force_compute(sophomore_databox);
}
} // namespace sophomore

namespace junior {
/// [tagged_tuple_tags]
struct Velocity {
  using type = double;
};
struct Radius {
  using type = double;
};
struct Density {
  using type = double;
};
struct Volume {
  using type = double;
};
/// [tagged_tuple_tags]

double tagged_tuple_databox_1() noexcept {
/// [tagged_tuple_databox_1]
tuples::TaggedTuple<Velocity, Radius, Density, Volume> junior_databox{
  4.0, 2.0, 0.5, 10.0};
/// [tagged_tuple_databox_1]
  return tuples::get<Density>(junior_databox) *
         tuples::get<Volume>(junior_databox) *
         tuples::get<Velocity>(junior_databox) *
         tuples::get<Velocity>(junior_databox) /
         tuples::get<Radius>(junior_databox);
}

/// [tagged_tuple_mass_compute]
template<typename... Tags>
double mass_compute(const tuples::TaggedTuple<Tags...>& box) noexcept {
  return tuples::get<Density>(box) * tuples::get<Volume>(box);
}
/// [tagged_tuple_mass_compute]

/// [tagged_tuple_acceleration_compute]
template<typename... Tags>
double acceleration_compute(const tuples::TaggedTuple<Tags...>& box) noexcept {
  return tuples::get<Velocity>(box) * tuples::get<Velocity>(box) /
         tuples::get<Radius>(box);
}
/// [tagged_tuple_acceleration_compute]

/// [tagged_tuple_force_compute]
template<typename... Tags>
double force_compute(const tuples::TaggedTuple<Tags...>& box) noexcept {
  const double mass = mass_compute(box);
  const double acceleration = acceleration_compute(box);
  return mass * acceleration;
}
/// [tagged_tuple_force_compute]
} // namespace junior

namespace proper{
/// [proper_databox_tags]
struct Velocity : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Velocity"; }
};
struct Radius : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Radius"; }
};
struct Density : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Density"; }
};
struct Volume : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Volume"; }
};
struct Mass : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Mass"; }
};
/// [proper_databox_tags]

double refined_databox_1() noexcept {
/// [refined_databox]
const auto refined_databox = db::create<
    db::AddSimpleTags<
      Velocity, Radius, Density, Volume>>(4.0, 2.0, 0.5, 10.0);
/// [refined_databox]
/// [refined_databox_get]
const double velocity = db::get<Velocity>(refined_databox);
/// [refined_databox_get]
const double radius = db::get<Radius>(refined_databox);
const double density = db::get<Density>(refined_databox);
const double volume = db::get<Volume>(refined_databox);
return density * volume *  velocity * velocity / radius;
}

double mass_from_density_and_volume(
  const double& density, const double& volume) noexcept {
  return density * volume;
}
/// [compute_tags]
struct MassCompute : db::ComputeTag, Mass {
  static std::string name() noexcept { return "MassCompute"; }
  static constexpr auto function = &mass_from_density_and_volume;
  using argument_tags = tmpl::list<Density, Volume>;
};
/// [compute_tags]

double acceleration_from_velocity_and_radius(
  const double& velocity, const double& radius) noexcept {
  return velocity * velocity / radius;
}

struct Acceleration : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Acceleration"; }
};

struct AccelerationCompute : db::ComputeTag, Acceleration {
  static std::string name() noexcept { return "AccelerationCompute"; }
  static constexpr auto function = &acceleration_from_velocity_and_radius;
  using argument_tags = tmpl::list<Velocity, Radius>;
};

/// [compute_tags_force_compute]
struct Force : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Force"; }
};

struct ForceCompute : db::ComputeTag, Force {
  static std::string name() noexcept { return "ForceCompute"; }
  static constexpr auto function(
    const double& mass, const double& acceleration) noexcept {
    return mass * acceleration; }
  using argument_tags = tmpl::list<Mass, Acceleration>;
};
/// [compute_tags_force_compute]
} // namespace proper

/// [mutate_tags]
struct Time : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "Time"; }
};

struct TimeStep : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "TimeStep"; }
};

struct EarthGravity : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "EarthGravity"; }
};

struct FallingSpeed : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "FallingSpeed"; }
};
/// [mutate_tags]

/// [intended_mutation]
struct IntendedMutation {
  static void apply(const gsl::not_null<double*> time,
     const gsl::not_null<double*> falling_speed,
     const double time_step,
     const double earth_gravity) {
    *time += time_step;
    *falling_speed += time_step * earth_gravity;
  }
};
/// [intended_mutation]

/// [intended_mutation2]
struct IntendedMutation2 {
  using return_tags = tmpl::list<Time, FallingSpeed>;
  using argument_tags = tmpl::list<TimeStep, EarthGravity>;

  static void apply(const gsl::not_null<double*> time,
     const gsl::not_null<double*> falling_speed,
     const double time_step,
     const double earth_gravity) {
    *time += time_step;
    *falling_speed += time_step * earth_gravity;
  }
};
/// [intended_mutation2]

/// [my_first_action]
template <typename Mutator>
struct MyFirstAction{
  template<typename DbTagsList>
  static void apply(
    const gsl::not_null<db::DataBox<DbTagsList>*> time_dependent_databox)
      noexcept {
    db::mutate_apply<Mutator>(time_dependent_databox);
  }
};
/// [my_first_action]

} // namespace
// clang-format on

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Documentation",
                  "[Unit][DataStructures]") {
const double velocity = 4.0;
const double radius = 2.0;
const double density = 0.5;
const double volume = 10.0;
const double mass = mass_compute(density, volume);
const double acceleration = acceleration_compute(velocity, radius);
const double force = force_compute(velocity, radius, density, volume);
CHECK(mass == density*volume);
CHECK(acceleration == velocity * velocity / radius);
CHECK(force == mass * acceleration);
failed_acceleration();
CHECK(force == working_without_databoxes_1());
CHECK(force == working_without_databoxes_2());
CHECK(force == std_map_databox_1());
CHECK(force == std_map_databox_2());
std::map<std::string, double> naive_databox;
naive_databox["Velocity"] = velocity;
naive_databox["Radius"] = radius;
naive_databox["Density"] = density;
naive_databox["Volume"] = volume;
CHECK(mass == mass_compute(naive_databox));
CHECK(acceleration == acceleration_compute(naive_databox));
CHECK(force == force_compute(naive_databox));
CHECK(std_tuple_databox_example());
std::tuple<std::pair<sophomore::Velocity, double>,
           std::pair<sophomore::Radius, double>,
           std::pair<sophomore::Density, double>,
           std::pair<sophomore::Volume, double>>
    sophomore_databox =
        std::make_tuple(std::make_pair(sophomore::Velocity{}, 4.0),
                        std::make_pair(sophomore::Radius{}, 2.0),
                        std::make_pair(sophomore::Density{}, 0.5),
                        std::make_pair(sophomore::Volume{}, 10.0));
CHECK(mass == sophomore::mass_compute(sophomore_databox));
CHECK(acceleration == sophomore::acceleration_compute(sophomore_databox));
CHECK(force == sophomore::force_compute(sophomore_databox));
CHECK(force == sophomore::std_tuple_databox_1());
CHECK(force == sophomore::std_tuple_databox_2());
tuples::TaggedTuple<junior::Velocity, junior::Radius, junior::Density,
                    junior::Volume>
    junior_databox{velocity, radius, density, volume};
CHECK(mass == junior::mass_compute(junior_databox));
CHECK(acceleration == junior::acceleration_compute(junior_databox));
CHECK(force == junior::force_compute(junior_databox));
CHECK(force == junior::tagged_tuple_databox_1());
CHECK(force == proper::refined_databox_1());
const auto refined_databox = db::create<
    db::AddSimpleTags<proper::Velocity, proper::Radius, proper::Density,
                      proper::Volume>,
    db::AddComputeTags<proper::MassCompute, proper::AccelerationCompute,
                       proper::ForceCompute>>(velocity, radius, density,
                                              volume);
CHECK(force == db::get<proper::Force>(refined_databox));

/// [time_dep_databox]
auto time_dependent_databox = db::create<
    db::AddSimpleTags<
      Time, TimeStep, EarthGravity, FallingSpeed>>(0.0, 0.1, -9.8, -10.0);
db::mutate_apply<
  //MutateTags
  tmpl::list<Time, FallingSpeed>,
  //ArgumentTags
  tmpl::list<TimeStep, EarthGravity>>(
  [](const gsl::not_null<double*> time,
     const gsl::not_null<double*> falling_speed,
     const double& time_step,
     const double& earth_gravity) {
    *time += time_step;
    *falling_speed += time_step * earth_gravity;
  },
  make_not_null(&time_dependent_databox));
/// [time_dep_databox]
CHECK(0.0 + 0.1 == approx(db::get<Time>(time_dependent_databox)));
CHECK(-10.0 + 0.1 * -9.8 ==
  approx(db::get<FallingSpeed>(time_dependent_databox)));

/// [time_dep_databox2]
db::mutate_apply<
  tmpl::list<Time, FallingSpeed>,
  tmpl::list<TimeStep, EarthGravity>>(
    IntendedMutation{}, make_not_null(&time_dependent_databox));
/// [time_dep_databox2]
CHECK(0.0 + 0.2 == approx(db::get<Time>(time_dependent_databox)));
CHECK(-10.0 + 0.2 * -9.8 ==
  approx(db::get<FallingSpeed>(time_dependent_databox)));

/// [time_dep_databox3]
db::mutate_apply<IntendedMutation2>(make_not_null(&time_dependent_databox));
/// [time_dep_databox3]
CHECK(0.0 + 0.3 == approx(db::get<Time>(time_dependent_databox)));
CHECK(-10.0 + 0.3 * -9.8 ==
  approx(db::get<FallingSpeed>(time_dependent_databox)));

/// [time_dep_databox4]
MyFirstAction<IntendedMutation2>::apply(make_not_null(&time_dependent_databox));
/// [time_dep_databox4]
CHECK(0.0 + 0.4 == approx(db::get<Time>(time_dependent_databox)));
CHECK(-10.0 + 0.4 * -9.8 ==
  approx(db::get<FallingSpeed>(time_dependent_databox)));
}
