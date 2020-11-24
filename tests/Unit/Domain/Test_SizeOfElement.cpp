// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_1d() noexcept {
  INFO("1d");
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.2));
  const ElementId<1> element_id(0, {{{2, 3}}});
  const ElementMap<1, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  // for this affine map, expected size = width of block / number of elements
  const auto size_expected = make_array<1>(0.225);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

void test_2d() noexcept {
  INFO("2d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(0, {{{1, 1}, {2, 0}}});
  const ElementMap<2, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  const auto size_expected = make_array(0.05, 0.425);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

void test_3d() noexcept {
  INFO("3d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2),
               Affine(-1.0, 1.0, 12.0, 12.5)},
      domain::CoordinateMaps::Rotation<3>(0.7, 2.3, -0.4));
  const ElementId<3> element_id(0, {{{3, 5}, {1, 0}, {2, 3}}});
  const ElementMap<3, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  const auto size_expected = make_array(0.0125, 0.85, 0.125);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

template <size_t Dim>
using CubicScale = domain::CoordinateMaps::TimeDependent::CubicScale<Dim>;

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
make_single_expansion_functions_of_time() noexcept {
  const double initial_time = 0.0;
  const double expiration_time = 10.0;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time.insert(std::make_pair(
      "Expansion",
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, std::array<DataVector, 3>{{{0.0}, {1.0}, {0.0}}},
          expiration_time)));
  return functions_of_time;
}

void test_1d_moving_mesh(const std::array<double, 4>& times_to_check) noexcept {
  INFO("1d with moving mesh");
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.2));
  const ElementId<1> element_id(0, {{{2, 3}}});
  const ElementMap<1, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          CubicScale<1>{10.0, "Expansion", "Expansion"});
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  for (const double time : times_to_check) {
    const auto element_size = size_of_element(
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);
    // for this affine map, expected size = width of block / number of
    // elements
    const auto expected_size_of_element = make_array<1>(0.225 * time);
    CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
  }
}

void test_2d_moving_mesh(const std::array<double, 4>& times_to_check) noexcept {
  INFO("2d with moving mesh");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(0, {{{1, 1}, {2, 0}}});
  const ElementMap<2, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          CubicScale<2>{10.0, "Expansion", "Expansion"});
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  for (const double time : times_to_check) {
    const auto element_size = size_of_element(
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);
    const auto expected_size_of_element = make_array(0.05, 0.425) * time;
    CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
  }
}

void test_3d_moving_mesh(const std::array<double, 4>& times_to_check) noexcept {
  INFO("3d with moving mesh");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine3D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2),
               Affine(-1.0, 1.0, 12.0, 12.5)},
      domain::CoordinateMaps::Rotation<3>(0.7, 2.3, -0.4));
  const ElementId<3> element_id(0, {{{3, 5}, {1, 0}, {2, 3}}});
  const ElementMap<3, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const auto grid_to_inertial_map =
      domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
          CubicScale<3>{10.0, "Expansion", "Expansion"});
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  for (const double time : times_to_check) {
    const auto element_size = size_of_element(
        logical_to_grid_map, grid_to_inertial_map, time, functions_of_time);
    const auto expected_size_of_element =
        make_array(0.0125, 0.85, 0.125) * time;
    CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
  }
}

void test_compute_tag(const std::array<double, 4>& times_to_check) noexcept {
  INFO("compute tag in 2d, with moving mesh");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(0, {{{1, 1}, {2, 0}}});
  ElementMap<2, Frame::Grid> logical_to_grid_map(element_id, std::move(map));
  auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          CubicScale<2>{10.0, "Expansion", "Expansion"});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  auto box =
      db::create<db::AddSimpleTags<domain::Tags::ElementMap<2, Frame::Grid>,
                                   domain::CoordinateMaps::Tags::CoordinateMap<
                                       2, Frame::Grid, Frame::Inertial>,
                                   ::Tags::Time, domain::Tags::FunctionsOfTime>,
                 db::AddComputeTags<domain::Tags::SizeOfElementCompute<2>>>(
          std::move(logical_to_grid_map), std::move(grid_to_inertial_map), 0.0,
          std::move(functions_of_time));

  for (const double time : times_to_check) {
    db::mutate<::Tags::Time>(
        make_not_null(&box),
        [time](const gsl::not_null<double*> time_ptr) noexcept {
          *time_ptr = time;
        });
    const auto expected_size_of_element = make_array(0.05, 0.425) * time;
    CHECK_ITERABLE_APPROX(db::get<domain::Tags::SizeOfElement<2>>(box),
                          expected_size_of_element);
  }
}

template <size_t Dim>
void test_tags() {
  TestHelpers::db::test_simple_tag<domain::Tags::SizeOfElement<Dim>>(
      "SizeOfElement");
  TestHelpers::db::test_compute_tag<domain::Tags::SizeOfElementCompute<Dim>>(
      "SizeOfElement");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.SizeOfElement", "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();

  const std::array<double, 4> times_to_check{{0.0, 0.3, 1.1, 7.8}};
  test_1d_moving_mesh(times_to_check);
  test_2d_moving_mesh(times_to_check);
  test_3d_moving_mesh(times_to_check);
  test_compute_tag(times_to_check);

  test_tags<1>();
  test_tags<2>();
  test_tags<3>();
}
