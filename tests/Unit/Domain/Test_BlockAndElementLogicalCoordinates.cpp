// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim, typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates(
    const Domain<Dim, TargetFrame>& domain,
    const std::vector<std::array<size_t, Dim>>& refinement_levels,
    const size_t n_pts) noexcept {
  const auto all_element_ids = initial_element_ids<Dim>(refinement_levels);

  // Random element_id for each point.
  const auto element_ids = [&all_element_ids, &n_pts ]() noexcept {
    std::uniform_int_distribution<size_t> ran(0, all_element_ids.size()-1);
    std::mt19937 gen;
    std::vector<ElementId<Dim>> ids(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      ids[s] = all_element_ids[ran(gen)];
    }
    return ids;
  }
  ();
  CAPTURE_PRECISE(element_ids);

  // Random element logical coords for each point
  const auto element_coords = [&n_pts]() noexcept {
    // Assumes logical coords go from -1 to 1.
    std::uniform_real_distribution<double> ran(-1.0, 1.0);
    std::mt19937 gen;
    std::vector<tnsr::I<double, Dim, Frame::Logical>> coords(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      for (size_t d = 0; d < Dim; ++d) {
        coords[s].get(d) = ran(gen);
      }
    }
    return coords;
  }
  ();
  CAPTURE_PRECISE(element_coords);

  // Compute expected map of element_ids to ElementLogicalCoordHolders.
  // This is just re-organizing and re-bookkeeping element_ids and
  // element_coords into the same structure that will be returned by
  // the function we are testing.
  const auto expected_coord_holders =
      [&element_ids, &element_coords, &n_pts ]() noexcept {
    // This is complicated because we don't know ahead of time
    // how many points are in each element.  So we do a first pass
    // filling a structure that you can easily push_back to.
    struct coords_plus_offset {
      std::vector<std::array<double, Dim>> coords;
      std::vector<size_t> offsets;
    };
    std::unordered_map<ElementId<Dim>, coords_plus_offset> coords_plus_offsets;
    for (size_t s = 0; s < n_pts; ++s) {
      auto new_coords = make_array<Dim>(0.0);
      for (size_t d = 0; d < Dim; ++d) {
        gsl::at(new_coords, d) = element_coords[s].get(d);
      }
      auto pos = coords_plus_offsets.find(element_ids[s]);
      if (pos == coords_plus_offsets.end()) {
        coords_plus_offsets.emplace(element_ids[s],
                                    coords_plus_offset{{new_coords}, {{s}}});
      } else {
        pos->second.coords.push_back(new_coords);
        pos->second.offsets.push_back(s);
      }
    }

    // The second pass fills the desired structure.
    std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>> holders;
    for (const auto& coord_holder : coords_plus_offsets) {
      const size_t num_grid_pts = coord_holder.second.offsets.size();
      tnsr::I<DataVector, Dim, Frame::Logical> coords(num_grid_pts);
      for (size_t s = 0; s < num_grid_pts; ++s) {
        for (size_t d = 0; d < Dim; ++d) {
          coords.get(d)[s] = gsl::at(coord_holder.second.coords[s], d);
        }
      }
      holders.emplace(
          coord_holder.first,
          ElementLogicalCoordHolder<Dim>{coords, coord_holder.second.offsets});
    }
    return holders;
  }
  ();

  // Transform element_coords to frame coords
  const auto frame_coords =
      [&n_pts, &domain, &element_ids, &element_coords ]() noexcept {
    tnsr::I<DataVector, Dim, TargetFrame> coords(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      const auto& my_block = domain.blocks()[element_ids[s].block_id()];
      ElementMap<Dim, TargetFrame> map{element_ids[s],
                                       my_block.coordinate_map().get_clone()};
      const auto coord_one_point = map(element_coords[s]);
      for (size_t d = 0; d < Dim; ++d) {
        coords.get(d)[s] = coord_one_point.get(d);
      }
    }
    return coords;
  }
  ();

  const auto block_logical_result =
      block_logical_coordinates(domain, frame_coords);
  test_serialization(block_logical_result);

  for (size_t s = 0; s < n_pts; ++s) {
    CHECK(block_logical_result[s].id.get_index() == element_ids[s].block_id());
    // We don't know block logical coordinates here, so we can't
    // test them.
  }

  // Test versus all the element_ids.
  const auto element_logical_result =
      element_logical_coordinates(all_element_ids, block_logical_result);

  for (const auto& expected_holder_pair : expected_coord_holders) {
    const auto pos = element_logical_result.find(expected_holder_pair.first);
    CHECK(pos != element_logical_result.end());
    if (pos != element_logical_result.end()) {
      const auto& holder = pos->second;
      CHECK(holder.offsets == expected_holder_pair.second.offsets);
      CHECK_ITERABLE_APPROX(holder.element_logical_coords,
                            expected_holder_pair.second.element_logical_coords);
    }
  }

  // Make sure every element in element_logical_result is also
  // in expected_coord_holders.
  for (const auto& holder_pair : element_logical_result) {
    const auto pos = expected_coord_holders.find(holder_pair.first);
    CHECK(pos != expected_coord_holders.end());
  }
}

template <size_t Dim, typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates_unrefined(
    const Domain<Dim, TargetFrame>& domain, const size_t n_pts) noexcept {
  const size_t n_blocks = domain.blocks().size();

  // Random block_id for each point.
  const auto block_ids = [&n_pts, &n_blocks ]() noexcept {
    std::uniform_int_distribution<size_t> ran(0, n_blocks-1);
    std::mt19937 gen;
    std::vector<size_t> ids(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      ids[s] = ran(gen);
    }
    return ids;
  }
  ();
  CAPTURE_PRECISE(block_ids);

  // Random block logical coords for each point
  // (block logical coords == element logical coords)
  const auto block_coords = [&n_pts]() noexcept {
    // Assumes logical coords go from -1 to 1.
    std::uniform_real_distribution<double> ran(-1.0, 1.0);
    std::mt19937 gen;
    std::vector<tnsr::I<double, Dim, Frame::Logical>> coords(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      for (size_t d = 0; d < Dim; ++d) {
        coords[s].get(d) = ran(gen);
      }
    }
    return coords;
  }
  ();
  CAPTURE_PRECISE(block_coords);

  // Map to frame coords
  const auto frame_coords =
      [&n_pts, &domain, &block_ids, &block_coords ]() noexcept {
    tnsr::I<DataVector, Dim, TargetFrame> coords(n_pts);
    for (size_t s = 0; s < n_pts; ++s) {
      const auto coord_one_point =
          domain.blocks()[block_ids[s]].coordinate_map()(block_coords[s]);
      for (size_t d = 0; d < Dim; ++d) {
        coords.get(d)[s] = coord_one_point.get(d);
      }
    }
    return coords;
  }
  ();

  const auto block_logical_result =
      block_logical_coordinates(domain, frame_coords);
  test_serialization(block_logical_result);

  for (size_t s = 0; s < n_pts; ++s) {
    CHECK(block_logical_result[s].id.get_index() == block_ids[s]);
    CHECK_ITERABLE_APPROX(block_logical_result[s].data, block_coords[s]);
  }
}

template <typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates_shell(
    const size_t n_pts) noexcept {
  const auto shell =
      domain::creators::Shell<TargetFrame>(1.5, 2.5, 2, {{1, 1}}, true, 1.0);
  const auto domain = shell.create_domain();
  fuzzy_test_block_and_element_logical_coordinates_unrefined(domain, n_pts);
  fuzzy_test_block_and_element_logical_coordinates(
      domain, shell.initial_refinement_levels(), n_pts);
}

template <typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates3(
    const size_t n_pts) noexcept {
  Domain<3, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<3>{2, 2, 2},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}},
          {Index<3>{}}),
      corners_for_rectilinear_domains(Index<3>{2, 2, 2}));
  fuzzy_test_block_and_element_logical_coordinates_unrefined(domain, n_pts);
  fuzzy_test_block_and_element_logical_coordinates(domain,
                                                   {{{0, 1, 2}},
                                                    {{2, 2, 2}},
                                                    {{2, 1, 1}},
                                                    {{0, 2, 1}},
                                                    {{2, 2, 2}},
                                                    {{2, 2, 2}},
                                                    {{1, 2, 0}},
                                                    {{2, 0, 1}}},
                                                   n_pts);
}

template <typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates2(
    const size_t n_pts) noexcept {
  Domain<2, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<2>{2, 3},
          std::array<std::vector<double>, 2>{
              {{0.0, 0.5, 1.0}, {0.0, 0.33, 0.66, 1.0}}},
          {Index<2>{}}),
      corners_for_rectilinear_domains(Index<2>{2, 3}));
  fuzzy_test_block_and_element_logical_coordinates_unrefined(domain, n_pts);
  fuzzy_test_block_and_element_logical_coordinates(
      domain, {{{0, 1}}, {{2, 1}}, {{2, 2}}, {{3, 2}}, {{0, 0}}, {{2, 0}}},
      n_pts);
}

template <typename TargetFrame>
void fuzzy_test_block_and_element_logical_coordinates1(
    const size_t n_pts) noexcept {
  Domain<1, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<1>{2}, std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.0}}},
          {Index<1>{}}),
      corners_for_rectilinear_domains(Index<1>{2}));
  fuzzy_test_block_and_element_logical_coordinates_unrefined(domain, n_pts);
  fuzzy_test_block_and_element_logical_coordinates(domain, {{{0}}, {{3}}},
                                                   n_pts);
}

template <size_t Dim, typename TargetFrame>
void test_block_and_element_logical_coordinates(
    const Domain<Dim, TargetFrame>& domain,
    const std::vector<std::array<double, Dim>>& x_frame,
    const std::vector<size_t>& expected_block_ids,
    const std::vector<std::array<double, Dim>>& expected_block_logical,
    const std::vector<ElementId<Dim>>& element_ids,
    const std::vector<size_t>& expected_element_id_indices,
    const std::vector<std::vector<size_t>>& expected_offset,
    const std::vector<std::vector<std::array<double, Dim>>>&
        expected_element_logical) noexcept {
  tnsr::I<DataVector, Dim, TargetFrame> frame_coords(x_frame.size());
  std::vector<tnsr::I<double, Dim, Frame::Logical>> expected_logical_coords(
      x_frame.size());
  for (size_t s = 0; s < x_frame.size(); ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      frame_coords.get(d)[s] = gsl::at(x_frame[s], d);
      expected_logical_coords[s].get(d) = gsl::at(expected_block_logical[s], d);
    }
  }

  const auto block_logical_result =
      block_logical_coordinates(domain, frame_coords);
  for (size_t s = 0; s < x_frame.size(); ++s) {
    CHECK(block_logical_result[s].id.get_index() == expected_block_ids[s]);
    CHECK_ITERABLE_APPROX(block_logical_result[s].data,
                          expected_logical_coords[s]);
  }

  test_serialization(block_logical_result);

  const auto element_logical_result =
      element_logical_coordinates(element_ids, block_logical_result);

  std::vector<tnsr::I<DataVector, Dim, Frame::Logical>> expected_elem_logical;
  for (const auto& coord : expected_element_logical) {
    tnsr::I<DataVector, Dim, Frame::Logical> dum(coord.size());
    for (size_t s = 0; s < coord.size(); ++s) {
      for (size_t d = 0; d < Dim; ++d) {
        dum.get(d)[s] = gsl::at(coord[s], d);
      }
    }
    expected_elem_logical.emplace_back(std::move(dum));
  }

  std::vector<ElementId<Dim>> expected_ids;
  expected_ids.reserve(expected_element_id_indices.size());
  for (const auto& index : expected_element_id_indices) {
    expected_ids.push_back(element_ids[index]);
  }

  for (size_t s = 0; s < expected_ids.size(); ++s) {
    const auto pos = element_logical_result.find(expected_ids[s]);
    CHECK(pos != element_logical_result.end());
    if (pos != element_logical_result.end()) {
      const auto& holder = pos->second;
      CHECK(holder.offsets == expected_offset[s]);
      CHECK_ITERABLE_APPROX(holder.element_logical_coords,
                            expected_elem_logical[s]);
    }
  }
  // Make sure we got all the elements
  for (const auto& holder_pair : element_logical_result) {
    const auto pos =
        std::find(expected_ids.begin(), expected_ids.end(), holder_pair.first);
    CHECK(pos != expected_ids.end());
  }
}

template <typename TargetFrame>
void test_block_and_element_logical_coordinates1() noexcept {
  Domain<1, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<1>{2}, std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.0}}},
          {Index<1>{}}),
      corners_for_rectilinear_domains(Index<1>{2}));

  std::vector<std::array<double, 1>> x_frame{
      {{0.1}},
      {{0.8}},
      {{0.5 + 1000.0 * std::numeric_limits<double>::epsilon()}},
      {{0.5}}};
  std::vector<size_t> expected_block_ids{0, 1, 1, 0};
  std::vector<std::array<double, 1>> expected_x_logical{
      {{-0.6}},
      {{0.2}},
      {{4000.0 * std::numeric_limits<double>::epsilon() - 1.0}},
      {{1.0}}
      // The last result is 1.0 because it checks the lower block_id first.
  };

  // Create some Elements.  I (Mark) did this by hand.
  auto element_ids = initial_element_ids<1>({{{2}}, {{3}}});

  // I (Mark) computed these expected quantities by hand, given the
  // points above and the choices of elements.
  std::vector<size_t> expected_id_indices{{0, 8, 4, 3}};
  std::vector<std::vector<size_t>> expected_offset{
      std::vector<size_t>{0}, std::vector<size_t>{1}, std::vector<size_t>{2},
      std::vector<size_t>{3}};
  std::vector<std::vector<std::array<double, 1>>> expected_elem_log{
      std::vector<std::array<double, 1>>{{{0.6}}},
      std::vector<std::array<double, 1>>{{{0.6}}},
      std::vector<std::array<double, 1>>{
          {{32000.0 * std::numeric_limits<double>::epsilon() - 1.0}}},
      std::vector<std::array<double, 1>>{{{1.0}}}};

  test_block_and_element_logical_coordinates(
      domain, x_frame, expected_block_ids, expected_x_logical, element_ids,
      expected_id_indices, expected_offset, expected_elem_log);
}

template <typename TargetFrame>
void test_block_logical_coordinates1fail() noexcept {
  Domain<1, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<1>{2}, std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.0}}},
          {Index<1>{}}),
      corners_for_rectilinear_domains(Index<1>{2}));

  std::vector<std::array<double, 1>> x_frame{
      {{0.1}}, {{1.1}}, {{-0.2}}, {{0.5}}};
  tnsr::I<DataVector, 1, TargetFrame> frame_coords(x_frame.size());
  for (size_t s = 0; s < x_frame.size(); ++s) {
    for (size_t d = 0; d < 1; ++d) {
      frame_coords.get(d)[s] = gsl::at(x_frame[s], d);
    }
  }
  const auto block_logical_result =
      block_logical_coordinates(domain, frame_coords);
}

template <typename TargetFrame>
void test_block_and_element_logical_coordinates3() noexcept {
  Domain<3, TargetFrame> domain(
      maps_for_rectilinear_domains<TargetFrame>(
          Index<3>{2, 2, 2},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}},
          {Index<3>{}}),
      corners_for_rectilinear_domains(Index<3>{2, 2, 2}));

  std::vector<std::array<double, 3>> x_frame{
      {{0.1, 0.1, 0.1}},   {{0.05, 0.05, 0.05}}, {{0.24, 0.24, 0.24}},
      {{0.9, 0.24, 0.24}}, {{0.24, 0.8, 0.24}},  {{0.9, 0.8, 0.24}},
      {{0.1, 0.1, 0.7}},   {{0.1, 0.8, 0.7}},    {{0.7, 0.2, 0.7}},
      {{0.9, 0.9, 0.9}},   {{0.5, 0.75, 1.0}}};
  // The last point above lies on the boundary of a block and of an
  // element.  block_logical_coordinates should pick the smallest
  // block_id that it lies on.
  std::vector<size_t> expected_block_ids{{0, 0, 0, 1, 2, 3, 4, 6, 5, 7, 6}};
  std::vector<std::array<double, 3>> expected_x_logical{
      {{-0.6, -0.6, -0.6}},  {{-0.8, -0.8, -0.8}},  {{-0.04, -0.04, -0.04}},
      {{0.6, -0.04, -0.04}}, {{-0.04, 0.2, -0.04}}, {{0.6, 0.2, -0.04}},
      {{-0.6, -0.6, -0.2}},  {{-0.6, 0.2, -0.2}},   {{-0.2, -0.2, -0.2}},
      {{0.6, 0.6, 0.6}},     {{1.0, 0.0, 1.0}}};

  // Create some Elements.  I (Mark) did this by hand.
  auto element_ids = initial_element_ids<3>(4, {{0, 1, 2}});
  auto element_ids_0 = initial_element_ids<3>(0, {{2, 2, 2}});
  auto element_ids_1 = initial_element_ids<3>(1, {{2, 1, 1}});
  auto element_ids_2 = initial_element_ids<3>(2, {{0, 2, 1}});
  auto element_ids_3 = initial_element_ids<3>(3, {{2, 2, 2}});
  auto element_ids_6 = initial_element_ids<3>(6, {{2, 2, 2}});
  auto element_ids_5 = initial_element_ids<3>(5, {{1, 2, 0}});
  auto element_ids_7 = initial_element_ids<3>(7, {{2, 0, 1}});
  std::copy(element_ids_0.begin(), element_ids_0.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_1.begin(), element_ids_1.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_2.begin(), element_ids_2.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_3.begin(), element_ids_3.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_6.begin(), element_ids_6.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_5.begin(), element_ids_5.end(),
            std::back_inserter(element_ids));
  std::copy(element_ids_7.begin(), element_ids_7.end(),
            std::back_inserter(element_ids));

  // I (Mark) computed these expected quantities by hand, given the
  // points above and the choices of elements.
  std::vector<size_t> expected_id_indices{
      {1, 8, 29, 84, 92, 153, 169, 225, 239, 215}};
  // The last point above is on an element boundary;
  // element_logical_coordinates should choose the first element in
  // the list of elements it is passed.
  std::vector<std::vector<size_t>> expected_offset{
      std::vector<size_t>{6}, std::vector<size_t>{0, 1}, std::vector<size_t>{2},
      std::vector<size_t>{3}, std::vector<size_t>{4},    std::vector<size_t>{5},
      std::vector<size_t>{7}, std::vector<size_t>{8},    std::vector<size_t>{9},
      std::vector<size_t>{10}};
  std::vector<std::vector<std::array<double, 3>>> expected_elem_log{
      std::vector<std::array<double, 3>>{{{-0.6, -0.2, 0.2}}},
      std::vector<std::array<double, 3>>{{{0.6, 0.6, 0.6}},
                                         {{-0.2, -0.2, -0.2}}},
      std::vector<std::array<double, 3>>{{{0.84, 0.84, 0.84}}},
      std::vector<std::array<double, 3>>{{{-0.6, 0.92, 0.92}}},
      std::vector<std::array<double, 3>>{{{-0.04, -0.2, 0.92}}},
      std::vector<std::array<double, 3>>{{{-0.6, -0.2, 0.84}}},
      std::vector<std::array<double, 3>>{{{0.6, -0.2, 0.2}}},
      std::vector<std::array<double, 3>>{{{0.6, 0.2, -0.2}}},
      std::vector<std::array<double, 3>>{{{-0.6, 0.6, 0.2}}},
      std::vector<std::array<double, 3>>{{{1.0, 1.0, 1.0}}}};

  test_block_and_element_logical_coordinates(
      domain, x_frame, expected_block_ids, expected_x_logical, element_ids,
      expected_id_indices, expected_offset, expected_elem_log);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.BlockAndElementLogicalCoords",
                  "[Domain][Unit]") {
  test_block_and_element_logical_coordinates1<Frame::Grid>();
  test_block_and_element_logical_coordinates3<Frame::Grid>();
  fuzzy_test_block_and_element_logical_coordinates3<Frame::Grid>(20);
  fuzzy_test_block_and_element_logical_coordinates2<Frame::Grid>(20);
  fuzzy_test_block_and_element_logical_coordinates1<Frame::Grid>(20);
  fuzzy_test_block_and_element_logical_coordinates1<Frame::Grid>(0);
  fuzzy_test_block_and_element_logical_coordinates_shell<Frame::Grid>(20);
}

// [[OutputRegex, Found points that are not in any block.:
// x_frame = \(T\(0\)=1\.1,T\(0\)=-0\.2\)]]
SPECTRE_TEST_CASE("Unit.Domain.BlockLogicalCoords.Fail", "[Domain][Unit]") {
  ERROR_TEST();
  test_block_logical_coordinates1fail<Frame::Grid>();
}
