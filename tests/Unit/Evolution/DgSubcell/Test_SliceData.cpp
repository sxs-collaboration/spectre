// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
namespace {
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
void check_slice(
    const Index<Dim>& volume_extents, const Index<Dim>& slice_extents,
    const Direction<Dim>& direction, const size_t fixed_index,
    const size_t ghost_slice, const size_t component_index,
    const Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>>& volume_vars,
    const DirectionMap<Dim, std::vector<double>>& sliced_data) {
  const size_t volume_offset = component_index * volume_extents.product();
  const size_t slice_offset = component_index * slice_extents.product();
  CAPTURE(volume_offset);
  CAPTURE(slice_offset);

  for (SliceIterator
           volume_it(volume_extents, direction.dimension(), fixed_index),
       slice_it(slice_extents, direction.dimension(), ghost_slice);
       volume_it; ++volume_it, ++slice_it) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(*(volume_vars.data() + volume_offset + volume_it.volume_offset()) ==
          sliced_data.at(direction)[slice_offset + slice_it.volume_offset()]);
  }
}

template <>
void check_slice<1>(
    const Index<1>& volume_extents, const Index<1>& slice_extents,
    const Direction<1>& direction, const size_t fixed_index,
    const size_t ghost_slice, const size_t component_index,
    const Variables<tmpl::list<Tags::Scalar, Tags::Vector<1>>>& volume_vars,
    const DirectionMap<1, std::vector<double>>& sliced_data) {
  const size_t volume_offset = component_index * volume_extents.product();
  const size_t slice_offset = component_index * slice_extents.product();
  CAPTURE(volume_offset);
  CAPTURE(slice_offset);
  for (SliceIterator volume_it(volume_extents, direction.dimension(),
                               fixed_index);
       volume_it; ++volume_it) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(*(volume_vars.data() + volume_offset + volume_it.volume_offset()) ==
          sliced_data.at(direction)[slice_offset + ghost_slice]);
  }
}

template <size_t Dim>
void test_slice_data(const std::optional<size_t> do_not_slice_in_direction) {
  CAPTURE(do_not_slice_in_direction.value_or(-1));
  for (size_t number_of_ghost_points = 1; number_of_ghost_points < 5;
       ++number_of_ghost_points) {
    for (size_t num_pts_1d = 5; num_pts_1d < 7; ++num_pts_1d) {
      const Index<Dim> extents{num_pts_1d};
      Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> volume_vars{
          extents.product()};
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      std::iota(volume_vars.data(), volume_vars.data() + volume_vars.size(),
                0.0);

      DirectionMap<Dim, bool> directions_to_slice{};
      for (const auto& direction : Direction<Dim>::all_directions()) {
        directions_to_slice[direction] = true;
      }

      if (do_not_slice_in_direction.has_value()) {
        directions_to_slice[gsl::at(Direction<Dim>::all_directions(),
                                    *do_not_slice_in_direction)] = false;
      }

      const auto sliced_data = subcell::slice_data(
          volume_vars, extents, number_of_ghost_points, directions_to_slice);
      for (const auto& direction : Direction<Dim>::all_directions()) {
        CAPTURE(direction);
        if (directions_to_slice.count(direction) == 1 and
            directions_to_slice.at(direction)) {
          REQUIRE(sliced_data.count(direction) == 1);
          Index<Dim> subcell_slice_extents = extents;
          subcell_slice_extents[direction.dimension()] = number_of_ghost_points;

          for (size_t component_index = 0;
               component_index < volume_vars.number_of_independent_components;
               ++component_index) {
            for (size_t ghost_slice = 0; ghost_slice < number_of_ghost_points;
                 ++ghost_slice) {
              const size_t fixed_index = direction.side() == Side::Lower
                                             ? ghost_slice
                                             : extents[direction.dimension()] +
                                                   ghost_slice -
                                                   number_of_ghost_points;
              CAPTURE(number_of_ghost_points);
              CAPTURE(num_pts_1d);
              CAPTURE(direction);
              CAPTURE(component_index);
              CAPTURE(ghost_slice);
              CAPTURE(fixed_index);
              CAPTURE(volume_vars);
              CAPTURE(subcell_slice_extents);
              CAPTURE(sliced_data.at(direction));
              check_slice(extents, subcell_slice_extents, direction,
                          fixed_index, ghost_slice, component_index,
                          volume_vars, sliced_data);
            }
          }
        } else {
          CHECK(sliced_data.count(direction) == 0);
        }
      }
    }
  }
}

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  test_slice_data<Dim>(std::nullopt);
  for (size_t i = 0; i < 2 * Dim; ++i) {
    test_slice_data<Dim>(i);
  }
}

// [[TimeOut, 8]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SliceData", "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace evolution::dg::subcell
