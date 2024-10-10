// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace domain::creators::bco {
template <bool IsCylindrical>
using ExpMapOptions =
    typename TimeDependentMapOptions<IsCylindrical>::ExpansionMapOptions;
template <bool IsCylindrical>
using RotMapOptions =
    typename TimeDependentMapOptions<IsCylindrical>::RotationMapOptions;
template <bool IsCylindrical>
using TransMapOptions =
    typename TimeDependentMapOptions<IsCylindrical>::TranslationMapOptions;
template <bool IsCylindrical>
using ShapeMapAOptions = typename TimeDependentMapOptions<
    IsCylindrical>::template ShapeMapOptions<domain::ObjectLabel::A>;
template <bool IsCylindrical>
using ShapeMapBOptions = typename TimeDependentMapOptions<
    IsCylindrical>::template ShapeMapOptions<domain::ObjectLabel::B>;
namespace {
// Test produce_all_maps for 1-4 maps
using Expansion = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<3>;
using Shape = domain::CoordinateMaps::TimeDependent::Shape;
using Identity = domain::CoordinateMaps::Identity<3>;

using combo_maps_1 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Expansion>;
using combo_maps_1_expected = tmpl::list<detail::gi_map<Expansion>>;
using combo_maps_2 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Expansion, Rotation>;
using combo_maps_2_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Expansion, Rotation>>;
// clang-format on
using combo_maps_3 = detail::produce_all_maps<Frame::Grid, Frame::Inertial,
                                              Shape, Expansion, Rotation>;
using combo_maps_3_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Shape>,
               detail::gi_map<Shape, Expansion>,
               detail::gi_map<Shape, Rotation>,
               detail::gi_map<Expansion, Rotation>,
               detail::gi_map<Shape, Expansion, Rotation>>;
// clang-format on
using combo_maps_4 =
    detail::produce_all_maps<Frame::Grid, Frame::Inertial, Shape, Expansion,
                             Rotation, Identity>;
using combo_maps_4_expected =
    // clang-format off
    tmpl::list<detail::gi_map<Expansion>,
               detail::gi_map<Rotation>,
               detail::gi_map<Shape>,
               detail::gi_map<Identity>,
               detail::gi_map<Shape, Expansion>,
               detail::gi_map<Shape, Rotation>,
               detail::gi_map<Shape, Identity>,
               detail::gi_map<Expansion, Rotation>,
               detail::gi_map<Expansion, Identity>,
               detail::gi_map<Rotation, Identity>,
               detail::gi_map<Shape, Expansion, Rotation>,
               detail::gi_map<Shape, Expansion, Identity>,
               detail::gi_map<Shape, Rotation, Identity>,
               detail::gi_map<Expansion, Rotation, Identity>,
               detail::gi_map<Shape, Expansion, Rotation, Identity>>;
// clang-format on

template <typename List1, typename List2>
constexpr bool check_maps_list_v =
    std::is_same_v<tmpl::list_difference<List1, List2>, tmpl::list<>>;

static_assert(check_maps_list_v<combo_maps_1, combo_maps_1_expected>);
static_assert(check_maps_list_v<combo_maps_1_expected, combo_maps_1>);
static_assert(check_maps_list_v<combo_maps_2, combo_maps_2_expected>);
static_assert(check_maps_list_v<combo_maps_2_expected, combo_maps_2>);
static_assert(check_maps_list_v<combo_maps_3, combo_maps_3_expected>);
static_assert(check_maps_list_v<combo_maps_3_expected, combo_maps_3>);
static_assert(check_maps_list_v<combo_maps_4, combo_maps_4_expected>);
static_assert(check_maps_list_v<combo_maps_4_expected, combo_maps_4>);

template <bool IsCylindrical>
void test(const bool include_expansion, const bool include_rotation,
          const bool include_translation, const bool include_shape_a,
          const bool include_shape_b, const bool transition_ends_at_cube_A,
          const bool transition_ends_at_cube_B) {
  CAPTURE(IsCylindrical);
  CAPTURE(include_expansion);
  CAPTURE(include_rotation);
  CAPTURE(include_translation);
  CAPTURE(include_shape_a);
  CAPTURE(include_shape_b);
  std::optional<ExpMapOptions<IsCylindrical>> exp_map_options{};
  std::optional<RotMapOptions<IsCylindrical>> rot_map_options{};
  std::optional<TransMapOptions<IsCylindrical>> trans_map_options{};
  std::optional<ShapeMapAOptions<IsCylindrical>> shape_map_a_options{};
  std::optional<ShapeMapBOptions<IsCylindrical>> shape_map_b_options{};

  const std::array<double, 2> exp_values{1.0, 0.0};
  const double exp_outer_boundary_velocity = -0.01;
  const double exp_outer_boundary_timescale = 25.0;
  if (include_expansion) {
    exp_map_options = ExpMapOptions<IsCylindrical>{
        exp_values, exp_outer_boundary_velocity, exp_outer_boundary_timescale};
  }

  const std::array<double, 3> angular_velocity{0.2, -0.4, 0.6};
  if (include_rotation) {
    rot_map_options = RotMapOptions<IsCylindrical>{angular_velocity};
  }

  const std::array<std::array<double, 3>, 3> translation_values = {
      std::array{1.0, 0.5, -1.0}, std::array{0.3, -0.1, -0.2},
      std::array{0.0, 0.0, 0.0}};
  if (include_translation) {
    trans_map_options = TransMapOptions<IsCylindrical>{translation_values};
  }

  const std::array<double, 3> size_A_values{0.9, 0.08, 0.007};
  const size_t l_max_A = 8;
  if (include_shape_a) {
    shape_map_a_options =
        IsCylindrical ? ShapeMapAOptions<IsCylindrical>{l_max_A, std::nullopt,
                                                        size_A_values}
                      : ShapeMapAOptions<IsCylindrical>{
                            l_max_A, std::nullopt, size_A_values,
                            transition_ends_at_cube_A};
  }

  const std::array<double, 3> size_B_values{-0.001, -0.02, -0.3};
  const size_t l_max_B = 10;
  if (include_shape_b) {
    shape_map_b_options =
        IsCylindrical ? ShapeMapBOptions<IsCylindrical>{l_max_B, std::nullopt,
                                                        size_B_values}
                      : ShapeMapBOptions<IsCylindrical>{
                            l_max_B, std::nullopt, size_B_values,
                            transition_ends_at_cube_B};
  }

  const double initial_time = 1.5;
  if ((not include_expansion) and (not include_rotation) and
      (not include_translation) and (not include_shape_a) and
      (not include_shape_b)) {
    CHECK_THROWS_WITH(
        (TimeDependentMapOptions<IsCylindrical>{
            initial_time, exp_map_options, rot_map_options, trans_map_options,
            shape_map_a_options, shape_map_b_options}),
        Catch::Matchers::ContainsSubstring(
            "Time dependent map options were specified, but all options "
            "were 'None'. If you don't want time dependent maps, specify "
            "'None' for the TimeDependentMapOptions. If you want time "
            "dependent maps, specify options for at least one map."));
    return;
  }

  TimeDependentMapOptions<IsCylindrical> time_dep_options{
      initial_time,      exp_map_options,     rot_map_options,
      trans_map_options, shape_map_a_options, shape_map_b_options};

  CHECK(time_dep_options.has_distorted_frame_options(domain::ObjectLabel::A) ==
        include_shape_a);
  CHECK(time_dep_options.has_distorted_frame_options(domain::ObjectLabel::B) ==
        include_shape_b);

  std::unordered_map<std::string, double> expiration_times{
      {TimeDependentMapOptions<IsCylindrical>::expansion_name, 10.0},
      {TimeDependentMapOptions<IsCylindrical>::rotation_name,
       std::numeric_limits<double>::infinity()},
      {TimeDependentMapOptions<IsCylindrical>::translation_name,
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions<IsCylindrical>::size_names, 0), 15.5},
      {gsl::at(TimeDependentMapOptions<IsCylindrical>::size_names, 1),
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions<IsCylindrical>::shape_names, 0),
       std::numeric_limits<double>::infinity()},
      {gsl::at(TimeDependentMapOptions<IsCylindrical>::shape_names, 1), 19.1}};

  using ExpFoT = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using ExpBdryFoT = domain::FunctionsOfTime::FixedSpeedCubic;
  using RotFoT = domain::FunctionsOfTime::QuaternionFunctionOfTime<3>;
  using TransFoT = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using SizeFoT = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  using ShapeFoT = ExpFoT;
  ExpFoT expansion{
      initial_time,
      std::array<DataVector, 3>{
          {{gsl::at(exp_values, 0)}, {gsl::at(exp_values, 1)}, {0.0}}},
      expiration_times.at(
          TimeDependentMapOptions<IsCylindrical>::expansion_name)};
  ExpBdryFoT expansion_outer_boundary{1.0, initial_time,
                                      exp_outer_boundary_velocity,
                                      exp_outer_boundary_timescale};
  RotFoT rotation{initial_time,
                  std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
                  std::array<DataVector, 4>{{{3, 0.0},
                                             {gsl::at(angular_velocity, 0),
                                              gsl::at(angular_velocity, 1),
                                              gsl::at(angular_velocity, 2)},
                                             {3, 0.0},
                                             {3, 0.0}}},
                  expiration_times.at(
                      TimeDependentMapOptions<IsCylindrical>::rotation_name)};
  TransFoT translation{
      initial_time,
      std::array<DataVector, 3>{{{gsl::at(translation_values, 0)[0],
                                  gsl::at(translation_values, 0)[1],
                                  gsl::at(translation_values, 0)[2]},
                                 {gsl::at(translation_values, 1)[0],
                                  gsl::at(translation_values, 1)[1],
                                  gsl::at(translation_values, 1)[2]},
                                 {gsl::at(translation_values, 2)[0],
                                  gsl::at(translation_values, 2)[1],
                                  gsl::at(translation_values, 2)[2]}}},
      expiration_times.at(
          TimeDependentMapOptions<IsCylindrical>::translation_name)};
  SizeFoT size_A{initial_time,
                 std::array<DataVector, 4>{{{gsl::at(size_A_values, 0)},
                                            {gsl::at(size_A_values, 1)},
                                            {gsl::at(size_A_values, 2)},
                                            {0.0}}},
                 expiration_times.at(gsl::at(
                     TimeDependentMapOptions<IsCylindrical>::size_names, 0))};
  SizeFoT size_B{initial_time,
                 std::array<DataVector, 4>{{{gsl::at(size_B_values, 0)},
                                            {gsl::at(size_B_values, 1)},
                                            {gsl::at(size_B_values, 2)},
                                            {0.0}}},
                 expiration_times.at(gsl::at(
                     TimeDependentMapOptions<IsCylindrical>::size_names, 1))};
  const DataVector shape_A_zeros{
      ylm::Spherepack::spectral_size(l_max_A, l_max_A), 0.0};
  const DataVector shape_B_zeros{
      ylm::Spherepack::spectral_size(l_max_B, l_max_B), 0.0};
  ShapeFoT shape_A{
      initial_time,
      std::array<DataVector, 3>{shape_A_zeros, shape_A_zeros, shape_A_zeros},
      expiration_times.at(
          gsl::at(TimeDependentMapOptions<IsCylindrical>::shape_names, 0))};
  ShapeFoT shape_B{
      initial_time,
      std::array<DataVector, 3>{shape_B_zeros, shape_B_zeros, shape_B_zeros},
      expiration_times.at(
          gsl::at(TimeDependentMapOptions<IsCylindrical>::shape_names, 1))};

  const std::array<std::array<double, 3>, 2> centers{
      std::array{5.0, 0.01, 0.02}, std::array{-5.0, -0.01, -0.02}};
  const std::optional<std::array<double, 3>> cube_a_center =
      IsCylindrical ? std::optional<std::array<double, 3>>{}
                    : std::array<double, 3>{{4.25, -0.3, 0.2}};
  const std::optional<std::array<double, 3>> cube_b_center =
      IsCylindrical ? std::optional<std::array<double, 3>>{}
                    : std::array<double, 3>{{-4.25, 0.4, -0.15}};
  const double domain_envelope_radius = 15.0;
  const double domain_outer_radius = 20.0;

  const auto anchors = create_grid_anchors(centers[0], centers[1]);
  CHECK(anchors.count("Center") == 1);
  CHECK(anchors.count("CenterA") == 1);
  CHECK(anchors.count("CenterB") == 1);
  CHECK(anchors.at("Center") ==
        tnsr::I<double, 3, Frame::Grid>(std::array{0.0, 0.0, 0.0}));
  CHECK(anchors.at("CenterA") == tnsr::I<double, 3, Frame::Grid>(centers[0]));
  CHECK(anchors.at("CenterB") == tnsr::I<double, 3, Frame::Grid>(centers[1]));

  for (const auto& [excise_A, excise_B] :
       cartesian_product(make_array(true, false), make_array(true, false))) {
    CAPTURE(excise_A);
    CAPTURE(excise_B);
    using RadiiType = std::optional<std::array<double, IsCylindrical ? 2 : 3>>;
    RadiiType inner_outer_radii_A{};
    RadiiType inner_outer_radii_B{};

      if constexpr (IsCylindrical) {
        inner_outer_radii_A = std::array{0.8, 3.2};
      } else {
        inner_outer_radii_A = std::array{0.8, 1.4, 3.2};
      }
      if constexpr (IsCylindrical) {
        inner_outer_radii_B = std::array{0.5, 2.1};
      } else {
        inner_outer_radii_B = std::array{0.5, 0.9, 2.1};
      }
      const auto build_maps = [&time_dep_options, &centers,
                               &inner_outer_radii_A, &inner_outer_radii_B,
                               &domain_envelope_radius, &domain_outer_radius,
                               &cube_a_center, &cube_b_center,
                               filled_A = not excise_A,
                               filled_B = not excise_B]() {
        time_dep_options.build_maps(centers, cube_a_center, cube_b_center,
                                    inner_outer_radii_A, inner_outer_radii_B,
                                    filled_A, filled_B, domain_envelope_radius,
                                    domain_outer_radius);
      };

      if constexpr (not IsCylindrical) {
        if (include_shape_a and
            (not excise_A and not transition_ends_at_cube_A)) {
          CHECK_THROWS_WITH(build_maps(),
                            Catch::Matchers::ContainsSubstring(
                                "If the object is filled, the transition must "
                                "end at the cube."));
          continue;
        }
        if (include_shape_a and not excise_A) {
          CHECK_THROWS_WITH(
              build_maps(),
              Catch::Matchers::ContainsSubstring(
                  "the centers of the inner and outer object are different"));
          continue;
        }
        if (include_shape_b and
            (not excise_B and not transition_ends_at_cube_B)) {
          CHECK_THROWS_WITH(build_maps(),
                            Catch::Matchers::ContainsSubstring(
                                "If the object is filled, the transition must "
                                "end at the cube."));
          continue;
        }
        if (include_shape_b and not excise_B) {
          CHECK_THROWS_WITH(
              build_maps(),
              Catch::Matchers::ContainsSubstring(
                  "the centers of the inner and outer object are different"));
          continue;
        }
      }

    // Now for each object, either we have included shape map options and
    // excision info, or we didn't include either. (i.e. include_shape_map_?
    // and excise_? are both true or both false) so it's safe to build the
    // maps now
    build_maps();

    using IncludeDistortedMapType = typename TimeDependentMapOptions<
        IsCylindrical>::IncludeDistortedMapType;
    IncludeDistortedMapType include_distorted_map_A{};
    IncludeDistortedMapType include_distorted_map_B{};
    if constexpr (IsCylindrical) {
      include_distorted_map_A = include_shape_a;
      include_distorted_map_B = include_shape_b;
    } else {
      include_distorted_map_A =
          include_shape_a ? std::make_optional(0_st) : std::nullopt;
      include_distorted_map_B =
          include_shape_b ? std::make_optional(0_st) : std::nullopt;
    }

    if ((not include_rotation) and (not include_expansion) and
        (not include_translation) and (not include_shape_a)) {
      CHECK(time_dep_options
                .template grid_to_inertial_map<domain::ObjectLabel::A>(
                    include_distorted_map_A, true) == nullptr);
      continue;
    }
    if ((not include_rotation) and (not include_expansion) and
        (not include_translation) and (not include_shape_b)) {
      CHECK(time_dep_options
                .template grid_to_inertial_map<domain::ObjectLabel::B>(
                    include_distorted_map_B, true) == nullptr);
      continue;
    }

    const auto grid_to_distorted_map_A =
        time_dep_options.template grid_to_distorted_map<domain::ObjectLabel::A>(
            include_distorted_map_A);
    const auto grid_to_distorted_map_B =
        time_dep_options.template grid_to_distorted_map<domain::ObjectLabel::B>(
            include_distorted_map_B);
    const auto grid_to_inertial_map_A =
        time_dep_options.template grid_to_inertial_map<domain::ObjectLabel::A>(
            include_distorted_map_A, true);
    const auto grid_to_inertial_map_B =
        time_dep_options.template grid_to_inertial_map<domain::ObjectLabel::B>(
            include_distorted_map_B, true);
    // Even though the distorted to inertial map is not tied to a specific
    // object, we use `excise_?` to determine if the distorted map is
    // included just for testing.
    const auto distorted_to_inertial_map_A =
        time_dep_options
            .template distorted_to_inertial_map<domain::ObjectLabel::A>(
                include_distorted_map_A, true);
    const auto distorted_to_inertial_map_B =
        time_dep_options
            .template distorted_to_inertial_map<domain::ObjectLabel::B>(
                include_distorted_map_B, true);

    // All of these maps are tested individually. Rather than going through
    // the effort of coming up with a source coordinate and calculating
    // analytically what we would get after it's mapped, we just check whether
    // it's supposed to be a nullptr and if it's not that it's not the
    // identity and that the jacobians are time dependent.
    const auto check_map = [](const auto& map, const bool is_null,
                              const bool is_identity) {
      if (is_null) {
        CHECK(map == nullptr);
      } else {
        REQUIRE(map != nullptr);
        CHECK(map->is_identity() == is_identity);
        CHECK(map->inv_jacobian_is_time_dependent() != is_identity);
        CHECK(map->jacobian_is_time_dependent() != is_identity);
      }
    };
    {
      INFO("grid_to_distorted_map_A");
      check_map(grid_to_distorted_map_A, not include_shape_a, false);
    }
    {
      INFO("grid_to_distorted_map_B");
      check_map(grid_to_distorted_map_B, not include_shape_b, false);
    }
    {
      INFO("grid_to_inertial_map_A");
      check_map(grid_to_inertial_map_A, false, false);
    }
    {
      INFO("grid_to_inertial_map_B");
      check_map(grid_to_inertial_map_B, false, false);
    }
    {
      INFO("distorted_to_inertial_map_A");
      check_map(distorted_to_inertial_map_A, not include_shape_a,
                (not include_rotation) and (not include_expansion) and
                    (not include_translation) and include_shape_a);
    }
    {
      INFO("distorted_to_inertial_map_B");
      check_map(distorted_to_inertial_map_B, not include_shape_b,
                (not include_rotation) and (not include_expansion) and
                    (not include_translation) and include_shape_b);
    }
    // Test functions of time
    const auto functions_of_time =
        time_dep_options.create_functions_of_time(expiration_times);
    if (include_expansion) {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::expansion_name) == 1);
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<
                    IsCylindrical>::expansion_outer_boundary_name) == 1);
      CHECK(dynamic_cast<ExpFoT&>(
                *functions_of_time
                     .at(TimeDependentMapOptions<IsCylindrical>::expansion_name)
                     .get()) == expansion);
      CHECK(dynamic_cast<ExpBdryFoT&>(
                *functions_of_time
                     .at(TimeDependentMapOptions<
                         IsCylindrical>::expansion_outer_boundary_name)
                     .get()) == expansion_outer_boundary);
    } else {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::expansion_name) == 0);
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<
                    IsCylindrical>::expansion_outer_boundary_name) == 0);
    }
    if (include_rotation) {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::rotation_name) == 1);
      CHECK(dynamic_cast<RotFoT&>(
                *functions_of_time
                     .at(TimeDependentMapOptions<IsCylindrical>::rotation_name)
                     .get()) == rotation);
    } else {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::rotation_name) == 0);
    }
    if (include_translation) {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::translation_name) == 1);
      CHECK(
          dynamic_cast<TransFoT&>(
              *functions_of_time
                   .at(TimeDependentMapOptions<IsCylindrical>::translation_name)
                   .get()) == translation);
    } else {
      CHECK(functions_of_time.count(
                TimeDependentMapOptions<IsCylindrical>::translation_name) == 0);
    }
    if (include_shape_a) {
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::size_names, 0)) == 1);
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::shape_names, 0)) == 1);
      CHECK(dynamic_cast<SizeFoT&>(
                *functions_of_time
                     .at(gsl::at(
                         TimeDependentMapOptions<IsCylindrical>::size_names, 0))
                     .get()) == size_A);
      CHECK(
          dynamic_cast<ShapeFoT&>(
              *functions_of_time
                   .at(gsl::at(
                       TimeDependentMapOptions<IsCylindrical>::shape_names, 0))
                   .get()) == shape_A);
    } else {
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::size_names, 0)) == 0);
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::shape_names, 0)) == 0);
    }
    if (include_shape_b) {
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::size_names, 1)) == 1);
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::shape_names, 1)) == 1);
      CHECK(dynamic_cast<SizeFoT&>(
                *functions_of_time
                     .at(gsl::at(
                         TimeDependentMapOptions<IsCylindrical>::size_names, 1))
                     .get()) == size_B);
      CHECK(
          dynamic_cast<ShapeFoT&>(
              *functions_of_time
                   .at(gsl::at(
                       TimeDependentMapOptions<IsCylindrical>::shape_names, 1))
                   .get()) == shape_B);
    } else {
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::size_names, 1)) == 0);
      CHECK(functions_of_time.count(gsl::at(
                TimeDependentMapOptions<IsCylindrical>::shape_names, 1)) == 0);
    }
  }
}

template <bool IsCylindrical>
void check_names() {
  INFO("Check names");
  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions<IsCylindrical>::expansion_name == "Expansion"s);
  CHECK(TimeDependentMapOptions<IsCylindrical>::expansion_outer_boundary_name ==
        "ExpansionOuterBoundary"s);
  CHECK(TimeDependentMapOptions<IsCylindrical>::rotation_name == "Rotation"s);
  CHECK(TimeDependentMapOptions<IsCylindrical>::translation_name ==
        "Translation"s);
  CHECK(TimeDependentMapOptions<IsCylindrical>::size_names ==
        std::array{"SizeA"s, "SizeB"s});
  CHECK(TimeDependentMapOptions<IsCylindrical>::shape_names ==
        std::array{"ShapeA"s, "ShapeB"s});
}

template <bool IsCylindrical>
void test_errors() {
  INFO("Test errors");
  CAPTURE(IsCylindrical);
  CHECK_THROWS_WITH(
      (TimeDependentMapOptions<IsCylindrical>{
          1.0, std::nullopt, std::nullopt, std::nullopt,
          ShapeMapAOptions<IsCylindrical>{1, {}},
          ShapeMapBOptions<IsCylindrical>{8, {}}}),
      Catch::Matchers::ContainsSubstring("Initial LMax for object"));
  CHECK_THROWS_WITH(
      (TimeDependentMapOptions<IsCylindrical>{
          1.0, std::nullopt, std::nullopt, std::nullopt,
          ShapeMapAOptions<IsCylindrical>{6, {}},
          ShapeMapBOptions<IsCylindrical>{0, {}}}),
      Catch::Matchers::ContainsSubstring("Initial LMax for object"));
  CHECK_THROWS_WITH((TimeDependentMapOptions<IsCylindrical>{
                        1.0, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt}),
                    Catch::Matchers::ContainsSubstring(
                        "Time dependent map options were "
                        "specified, but all options were 'None'."));
  using RadiiType = std::array<double, IsCylindrical ? 2 : 3>;
  RadiiType radii{};
  if constexpr (IsCylindrical) {
    radii = std::array{0.1, 1.0};
  } else {
    radii = std::array{0.1, 0.5, 1.0};
  }
  if (IsCylindrical) {
    CHECK_THROWS_WITH(
        ([&radii]() {
          TimeDependentMapOptions<IsCylindrical> time_dep_opts{
              1.0,
              ExpMapOptions<IsCylindrical>{{1.0, 0.0}, 0.0, 0.01},
              RotMapOptions<IsCylindrical>{{0.0, 0.0, 0.0}},
              std::nullopt,
              ShapeMapAOptions<IsCylindrical>{8, {}},
              std::nullopt};
          time_dep_opts.build_maps(
              std::array{std::array{5.0, 0.0, 0.0}, std::array{-5.0, 0.0, 0.0}},
              {{7.5, 0.0, 0.0}}, {{-7.5, 0.0, 0.0}}, radii, radii, false, false,
              25.0, 100.0);
        }()),
        Catch::Matchers::ContainsSubstring(
            "When using the CylindricalBinaryCompactObject domain creator, "
            "the excision centers cannot be offset."));
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      TimeDependentMapOptions<IsCylindrical>{}.has_distorted_frame_options(
          domain::ObjectLabel::None),
      Catch::Matchers::ContainsSubstring(
          "object label for TimeDependentMapOptions must be either A or B"));
#endif
}

void test_worldtube_fots() {
  const std::array<double, 3> size_a_opts{{1.3, 1.2, 1.1}};
  const std::array<double, 3> size_b_opts{{1.4, 1.5, 1.6}};
  const double initial_time = 0.0;

  const double initial_expansion = 1.2;
  const double initial_expansion_deriv = 2.2;

  const TimeDependentMapOptions<false> worldtube_options{
      initial_time,
      ExpMapOptions<false>{
          {initial_expansion, initial_expansion_deriv}, 1., 1.},
      RotMapOptions<false>{{0.0, 0.0, 1.0}},
      std::nullopt,
      ShapeMapAOptions<false>{2, {}, std::make_optional(size_a_opts)},
      ShapeMapBOptions<false>{2, {}, std::make_optional(size_b_opts)}};
  const auto fots = worldtube_options.create_functions_of_time<true>({});
  CHECK(not fots.contains("Translation"));

  CHECK(fots.contains("Rotation"));
  const auto& rotation_fot = fots.at("Rotation");
  CHECK(dynamic_cast<domain::FunctionsOfTime::IntegratedFunctionOfTime*>(
      &*rotation_fot));
  CHECK(rotation_fot->time_bounds() ==
        std::array<double, 2>{{initial_time, initial_time + 1e-10}});
  const DataVector rotation_value{1., 0., 0., 0.};
  const DataVector rotation_deriv_value{0., 0., 0., 0.5};
  CHECK_ITERABLE_APPROX(rotation_fot->func(initial_time)[0], rotation_value);
  CHECK_ITERABLE_APPROX(rotation_fot->func_and_deriv(initial_time)[1],
                        rotation_deriv_value);

  CHECK(fots.contains("Expansion"));
  const auto& expansion_fot = fots.at("Expansion");
  CHECK(dynamic_cast<domain::FunctionsOfTime::IntegratedFunctionOfTime*>(
      &*expansion_fot));
  CHECK(expansion_fot->time_bounds() ==
        std::array<double, 2>{{initial_time, initial_time + 1e-10}});
  const DataVector expansion_value{initial_expansion};
  const DataVector expansion_deriv_value{initial_expansion_deriv};
  CHECK_ITERABLE_APPROX(expansion_fot->func(initial_time)[0], expansion_value);
  CHECK_ITERABLE_APPROX(expansion_fot->func_and_deriv(initial_time)[1],
                        expansion_deriv_value);

  CHECK(fots.contains("ExpansionOuterBoundary"));
  const auto& boundary_expansion = fots.at("ExpansionOuterBoundary");
  CHECK(dynamic_cast<domain::FunctionsOfTime::FixedSpeedCubic*>(
      &*boundary_expansion));
  CHECK(boundary_expansion->time_bounds() ==
        std::array<double, 2>{
            {initial_time, std::numeric_limits<double>::infinity()}});

  CHECK(fots.contains("SizeA"));
  const auto& size_a_fot = fots.at("SizeA");
  CHECK(dynamic_cast<domain::FunctionsOfTime::IntegratedFunctionOfTime*>(
      &*size_a_fot));
  CHECK(size_a_fot->time_bounds() ==
        std::array<double, 2>{{initial_time, initial_time + 1e-10}});
  const DataVector size_a_value{size_a_opts[0]};
  const DataVector size_a_deriv_value{size_a_opts[1]};
  CHECK_ITERABLE_APPROX(size_a_fot->func(initial_time)[0], size_a_value);
  CHECK_ITERABLE_APPROX(size_a_fot->func_and_deriv(initial_time)[1],
                        size_a_deriv_value);

  CHECK(fots.contains("SizeB"));
  const auto& size_b_fot = fots.at("SizeB");
  CHECK(dynamic_cast<domain::FunctionsOfTime::IntegratedFunctionOfTime*>(
      &*size_b_fot));
  CHECK(size_b_fot->time_bounds() ==
        std::array<double, 2>{{initial_time, initial_time + 1e-10}});
  const DataVector size_b_value{size_b_opts[0]};
  const DataVector size_b_deriv_value{size_b_opts[1]};
  CHECK_ITERABLE_APPROX(size_b_fot->func(initial_time)[0], size_b_value);
  CHECK_ITERABLE_APPROX(size_b_fot->func_and_deriv(initial_time)[1],
                        size_b_deriv_value);

  CHECK(fots.contains("ShapeA"));
  const auto& shape_a_fot = fots.at("ShapeA");
  CHECK(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
      &*shape_a_fot));
  CHECK(shape_a_fot->time_bounds() ==
        std::array<double, 2>{
            {initial_time, std::numeric_limits<double>::infinity()}});

  CHECK(fots.contains("ShapeB"));
  const auto& shape_b_fot = fots.at("ShapeB");
  CHECK(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
      &*shape_b_fot));
  CHECK(shape_b_fot->time_bounds() ==
        std::array<double, 2>{
            {initial_time, std::numeric_limits<double>::infinity()}});
}

}  // namespace

// [[TimeOut, 45]]
SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependentOptions.BinaryCompactObject",
    "[Domain][Unit]") {
  for (const auto& [include_expansion, include_rotation, include_translation,
                    include_shape_a, include_shape_b, transition_ends_at_cube_A,
                    transition_ends_at_cube_B] :
       cartesian_product(make_array(true, false), make_array(true, false),
                         make_array(true, false), make_array(true, false),
                         make_array(true, false), make_array(true, false),
                         make_array(true, false))) {
    test<false>(include_expansion, include_rotation, include_translation,
                include_shape_a, include_shape_b, transition_ends_at_cube_A,
                transition_ends_at_cube_B);
    // Filter out cases that are not valid for the cylindrical domain
    if (transition_ends_at_cube_A or transition_ends_at_cube_B) {
      continue;
    }
    test<true>(include_expansion, include_rotation, include_translation,
               include_shape_a, include_shape_b, transition_ends_at_cube_A,
               transition_ends_at_cube_B);
  }
  check_names<true>();
  check_names<false>();
  test_errors<true>();
  test_errors<false>();
  test_worldtube_fots();
}
}  // namespace domain::creators::bco
