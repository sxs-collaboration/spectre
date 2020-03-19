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
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
using CubicScale = domain::CoordMapsTimeDependent::CubicScale<Dim>;

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
make_single_expansion_functions_of_time() noexcept {
  const double initial_time = 0.0;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time.insert(std::make_pair(
      "Expansion",
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, std::array<DataVector, 3>{{{0.0}, {1.0}, {0.0}}})));
  return functions_of_time;
}

template <bool UseMovingMesh>
void test_1d(const std::array<double, 4>& times_to_check) noexcept {
  INFO("1d");
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.2));
  const ElementId<1> element_id(0,
                                std::array<SegmentId, 1>({{SegmentId(2, 3)}}));
  const ElementMap<1, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  const auto check_helper =
      [&functions_of_time, &logical_to_grid_map,
       &times_to_check](const auto& grid_to_inertial_map) noexcept {
        std::array<double, 1> element_size{};
        for (const double time : times_to_check) {
          size_of_element(make_not_null(&element_size), logical_to_grid_map,
                          grid_to_inertial_map, time, functions_of_time);
          // for this affine map, expected size = width of block / number of
          // elements
          const auto expected_size_of_element =
              make_array<1>(0.225 * (UseMovingMesh ? time : 1.0));
          CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
        }
      };

  if (UseMovingMesh) {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        CubicScale<1>{10.0, "Expansion", "Expansion"}));

  } else {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        domain::CoordinateMaps::Identity<1>{}));
  }
}

template <bool UseMovingMesh>
void test_2d(const std::array<double, 4>& times_to_check) noexcept {
  INFO("2d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(
      0, std::array<SegmentId, 2>({{SegmentId(1, 1), SegmentId(2, 0)}}));
  const ElementMap<2, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  const auto check_helper =
      [&functions_of_time, &logical_to_grid_map,
       &times_to_check](const auto& grid_to_inertial_map) noexcept {
        std::array<double, 2> element_size{};
        for (const double time : times_to_check) {
          size_of_element(make_not_null(&element_size), logical_to_grid_map,
                          grid_to_inertial_map, time, functions_of_time);
          const auto expected_size_of_element =
              make_array(0.05, 0.425) * (UseMovingMesh ? time : 1.0);
          CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
        }
      };

  if (UseMovingMesh) {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        CubicScale<2>{10.0, "Expansion", "Expansion"}));

  } else {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        domain::CoordinateMaps::Identity<2>{}));
  }
}

template <bool UseMovingMesh>
void test_3d(const std::array<double, 4>& times_to_check) noexcept {
  INFO("3d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine3D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2),
               Affine(-1.0, 1.0, 12.0, 12.5)},
      domain::CoordinateMaps::Rotation<3>(0.7, 2.3, -0.4));
  const ElementId<3> element_id(
      0, std::array<SegmentId, 3>(
             {{SegmentId(3, 5), SegmentId(1, 0), SegmentId(2, 3)}}));
  const ElementMap<3, Frame::Grid> logical_to_grid_map(element_id,
                                                       std::move(map));
  const std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time = make_single_expansion_functions_of_time();

  const auto check_helper =
      [&functions_of_time, &logical_to_grid_map,
       &times_to_check](const auto& grid_to_inertial_map) noexcept {
        std::array<double, 3> element_size{};
        for (const double time : times_to_check) {
          size_of_element(make_not_null(&element_size), logical_to_grid_map,
                          grid_to_inertial_map, time, functions_of_time);
          const auto expected_size_of_element =
              make_array(0.0125, 0.85, 0.125) * (UseMovingMesh ? time : 1.0);
          CHECK_ITERABLE_APPROX(element_size, expected_size_of_element);
        }
      };

  if (UseMovingMesh) {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        CubicScale<3>{10.0, "Expansion", "Expansion"}));

  } else {
    check_helper(domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
        domain::CoordinateMaps::Identity<3>{}));
  }
}

template <bool UseMovingMesh>
void test_compute_tag(const std::array<double, 4>& times_to_check) noexcept {
  INFO("compute tag");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Grid>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(
      0, std::array<SegmentId, 2>({{SegmentId(1, 1), SegmentId(2, 0)}}));
  ElementMap<2, Frame::Grid> logical_to_grid_map(element_id, std::move(map));
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 2>>
      grid_to_inertial_map = nullptr;
  if (UseMovingMesh) {
    grid_to_inertial_map =
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CubicScale<2>{10.0, "Expansion", "Expansion"});
  } else {
    grid_to_inertial_map =
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<2>{});
  }

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
    const auto expected_size_of_element =
        make_array(0.05, 0.425) * (UseMovingMesh ? time : 1.0);
    CHECK_ITERABLE_APPROX(db::get<domain::Tags::SizeOfElement<2>>(box),
                          expected_size_of_element);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.SizeOfElement", "[Domain][Unit]") {
  const std::array<double, 4> times_to_check{{0.0, 0.3, 1.1, 7.8}};

  test_1d<false>(times_to_check);
  test_2d<false>(times_to_check);
  test_3d<false>(times_to_check);
  test_compute_tag<false>(times_to_check);

  test_1d<true>(times_to_check);
  test_2d<true>(times_to_check);
  test_3d<true>(times_to_check);
  test_compute_tag<true>(times_to_check);
}
