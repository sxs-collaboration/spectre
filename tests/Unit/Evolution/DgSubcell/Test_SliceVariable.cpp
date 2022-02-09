// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <numeric>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceVariable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// types of Tensor to test
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct TensorI : db::SimpleTag {
  using type = ::tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
void test_slice(
    const Variables<tmpl::list<Tags::Scalar, Tags::TensorI<Dim>>>& volume_vars,
    const Index<Dim>& volume_extents, size_t num_ghost_pts,
    const Direction<Dim>& direction) {
  // retrieve a sliced Variables object
  auto sliced_vars = evolution::dg::subcell::slice_variable(
      volume_vars, volume_extents, num_ghost_pts, direction);

  Index<Dim> sliced_extents = volume_extents;
  sliced_extents[direction.dimension()] = num_ghost_pts;

  // for each ghost slices,
  for (size_t i_ghost = 0; i_ghost < num_ghost_pts; ++i_ghost) {
    // for each of Variables components ( Scalar x 1 + TensorI x Dim )
    for (size_t component_index = 0;
         component_index < volume_vars.number_of_independent_components;
         ++component_index) {
      const size_t fixed_index =
          direction.side() == Side::Lower
              ? i_ghost
              : volume_extents[direction.dimension()] + i_ghost - num_ghost_pts;

      const size_t volume_offset = component_index * volume_extents.product();
      const size_t slice_offset = component_index * sliced_extents.product();

      // check the result
      for (SliceIterator
               volume_it(volume_extents, direction.dimension(), fixed_index),
           slice_it(sliced_extents, direction.dimension(), i_ghost);
           volume_it; ++volume_it, ++slice_it) {
        CHECK(
            *(volume_vars.data() + volume_offset + volume_it.volume_offset()) ==
            *(sliced_vars.data() + slice_offset + slice_it.volume_offset()));
      }
    }
  }
}

template <size_t Dim>
void test() {
  for (size_t num_mesh_pts_1d = 3; num_mesh_pts_1d < 5; ++num_mesh_pts_1d) {
    // ghost zone size iterates up to the maximum possible value (mesh extent of
    // each dimension)
    for (size_t num_ghost_pts = 1; num_ghost_pts <= num_mesh_pts_1d;
         ++num_ghost_pts) {
      const Index<Dim> volume_extents{num_mesh_pts_1d};

      // Create volume tensors and assign values
      Variables<tmpl::list<Tags::Scalar, Tags::TensorI<Dim>>> volume_vars{
          volume_extents.product()};
      std::iota(volume_vars.data(), volume_vars.data() + volume_vars.size(),
                0.0);

      for (const auto& direction : Direction<Dim>::all_directions()) {
        // check slicing for each direction
        test_slice(volume_vars, volume_extents, num_ghost_pts, direction);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SliceVariable", "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
