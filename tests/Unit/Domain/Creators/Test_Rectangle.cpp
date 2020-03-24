// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/ProductMapsTimeDep.hpp"
#include "Domain/CoordinateMaps/ProductMapsTimeDep.tpp"
#include "Domain/CoordinateMaps/Translation.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeVector.hpp"

namespace domain {
namespace {
using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Translation = CoordMapsTimeDependent::Translation;
using Translation2D =
    CoordMapsTimeDependent::ProductOf2Maps<Translation, Translation>;

template <typename... FuncsOfTime>
void test_rectangle_construction(
    const creators::Rectangle& rectangle,
    const std::array<double, 2>& lower_bound,
    const std::array<double, 2>& upper_bound,
    const std::vector<std::array<size_t, 2>>& expected_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const std::vector<DirectionMap<2, BlockNeighbor<2>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<2>>>&
        expected_external_boundaries,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 2>>>&
        expected_grid_to_inertial_maps = {}) {
  const auto domain = rectangle.create_domain();

  CHECK(rectangle.initial_extents() == expected_extents);
  CHECK(rectangle.initial_refinement_levels() == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<
                  Frame::Logical,
                  tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                      Frame::Inertial, Frame::Grid>>(
          Affine2D{Affine{-1., 1., lower_bound[0], upper_bound[0]},
                   Affine{-1., 1., lower_bound[1], upper_bound[1]}})),
      10.0, rectangle.functions_of_time(), expected_grid_to_inertial_maps);
  test_initial_domain(domain, rectangle.initial_refinement_levels());
  TestHelpers::domain::creators::test_functions_of_time(
      rectangle, expected_functions_of_time);

  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  test_serialization(domain);
}

void test_rectangle() {
  INFO("Rectangle");
  const std::vector<std::array<size_t, 2>> grid_points{{{4, 6}}},
      refinement_level{{{3, 2}}};
  const std::array<double, 2> lower_bound{{-1.2, 3.0}}, upper_bound{{0.8, 5.0}};
  // default OrientationMap is aligned
  const OrientationMap<2> aligned_orientation{};

  const creators::Rectangle rectangle{lower_bound, upper_bound,
                                      std::array<bool, 2>{{false, false}},
                                      refinement_level[0], grid_points[0]};
  test_rectangle_construction(
      rectangle, lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{{}},
      std::vector<std::unordered_set<Direction<2>>>{
          {{Direction<2>::lower_xi()},
           {Direction<2>::upper_xi()},
           {Direction<2>::lower_eta()},
           {Direction<2>::upper_eta()}}});

  const creators::Rectangle periodic_x_rectangle{
      lower_bound, upper_bound, std::array<bool, 2>{{true, false}},
      refinement_level[0], grid_points[0]};
  test_rectangle_construction(
      periodic_x_rectangle, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::lower_xi(), {0, aligned_orientation}},
           {Direction<2>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<2>>>{
          {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}});

  const creators::Rectangle periodic_y_rectangle{
      lower_bound, upper_bound, std::array<bool, 2>{{false, true}},
      refinement_level[0], grid_points[0]};
  test_rectangle_construction(
      periodic_y_rectangle, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::lower_eta(), {0, aligned_orientation}},
           {Direction<2>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<2>>>{
          {{Direction<2>::lower_xi()}, {Direction<2>::upper_xi()}}});

  const creators::Rectangle periodic_xy_rectangle{
      lower_bound, upper_bound, std::array<bool, 2>{{true, true}},
      refinement_level[0], grid_points[0]};
  test_rectangle_construction(
      periodic_xy_rectangle, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::lower_xi(), {0, aligned_orientation}},
           {Direction<2>::upper_xi(), {0, aligned_orientation}},
           {Direction<2>::lower_eta(), {0, aligned_orientation}},
           {Direction<2>::upper_eta(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<2>>>{{}});
}

void test_rectangle_factory() {
  {
    INFO("Rectangle factory time independent");
    const auto domain_creator =
        TestHelpers::test_factory_creation<DomainCreator<2>>(
            "Rectangle:\n"
            "  LowerBound: [0,0]\n"
            "  UpperBound: [1,2]\n"
            "  IsPeriodicIn: [True,False]\n"
            "  InitialGridPoints: [3,4]\n"
            "  InitialRefinement: [2,3]\n");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_xi(), {0, {}}},
             {Direction<2>::upper_xi(), {0, {}}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}});
  }
  {
    INFO("Rectangle factory time dependent");
    const auto domain_creator =
        TestHelpers::test_factory_creation<DomainCreator<2>>(
            "Rectangle:\n"
            "  LowerBound: [0,0]\n"
            "  UpperBound: [1,2]\n"
            "  IsPeriodicIn: [True,False]\n"
            "  InitialGridPoints: [3,4]\n"
            "  InitialRefinement: [2,3]\n"
            "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 1.0\n"
            "      Velocity: [2.3, -0.3]\n"
            "      FunctionOfTimeNames: [TranslationX, TranslationY]");
    const auto* rectangle_creator =
        dynamic_cast<const creators::Rectangle*>(domain_creator.get());
    test_rectangle_construction(
        *rectangle_creator, {{0., 0.}}, {{1., 2.}}, {{{3, 4}}}, {{{2, 3}}},
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::lower_xi(), {0, {}}},
             {Direction<2>::upper_xi(), {0, {}}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {{Direction<2>::lower_eta()}, {Direction<2>::upper_eta()}}},
        std::make_tuple(
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationX",
                {1.0, std::array<DataVector, 3>{{{0.0}, {2.3}, {0.0}}}}},
            std::pair<std::string,
                      domain::FunctionsOfTime::PiecewisePolynomial<2>>{
                "TranslationY",
                {1.0, std::array<DataVector, 3>{{{0.0}, {-0.3}, {0.0}}}}}),
        make_vector_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation2D{Translation{"TranslationX"},
                          Translation{"TranslationY"}}));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Rectangle.Factory", "[Domain][Unit]") {
  test_rectangle();
  test_rectangle_factory();
}
}  // namespace domain
