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
#include "Evolution/DgSubcell/SliceTensor.hpp"
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

template <size_t Dim>
struct TensorIJ : db::SimpleTag {
  using type = ::tnsr::IJ<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
void test() {
  for (size_t num_mesh_pts_1d = 3; num_mesh_pts_1d < 5; ++num_mesh_pts_1d) {
    // ghost zone size iterates up to the maximum possible value (mesh extent of
    // each dimension)
    for (size_t num_ghost_pts = 1; num_ghost_pts <= num_mesh_pts_1d;
         ++num_ghost_pts) {
      const Index<Dim> volume_extents{num_mesh_pts_1d};

      // Create volume tensors and assign values
      Variables<
          tmpl::list<Tags::Scalar, Tags::TensorI<Dim>, Tags::TensorIJ<Dim>>>
          volume_vars{volume_extents.product()};
      std::iota(volume_vars.data(), volume_vars.data() + volume_vars.size(),
                0.0);

      auto& volume_scalar = get<Tags::Scalar>(volume_vars);
      auto& volume_tensor = get<Tags::TensorI<Dim>>(volume_vars);
      auto& volume_tensor_IJ = get<Tags::TensorIJ<Dim>>(volume_vars);

      for (const auto& direction : Direction<Dim>::all_directions()) {
        // slice data
        auto sliced_scalar = evolution::dg::subcell::slice_tensor_for_subcell(
            volume_scalar, volume_extents, num_ghost_pts, direction);
        auto sliced_tensor = evolution::dg::subcell::slice_tensor_for_subcell(
            volume_tensor, volume_extents, num_ghost_pts, direction);
        auto sliced_tensor_IJ =
            evolution::dg::subcell::slice_tensor_for_subcell(
                volume_tensor_IJ, volume_extents, num_ghost_pts, direction);

        Index<Dim> sliced_extents = volume_extents;
        sliced_extents[direction.dimension()] = num_ghost_pts;

        for (size_t i_ghost = 0; i_ghost < num_ghost_pts; ++i_ghost) {
          const size_t fixed_index =
              direction.side() == Side::Lower
                  ? i_ghost
                  : volume_extents[direction.dimension()] + i_ghost -
                        num_ghost_pts;

          // check the result
          for (SliceIterator volume_it(volume_extents, direction.dimension(),
                                       fixed_index),
               slice_it(sliced_extents, direction.dimension(), i_ghost);
               volume_it; ++volume_it, ++slice_it) {
            CHECK(*(get(volume_scalar).data() + volume_it.volume_offset()) ==
                  get(sliced_scalar)[slice_it.volume_offset()]);

            for (size_t i_tnsr = 0; i_tnsr < volume_tensor.size(); ++i_tnsr) {
              CHECK(*(volume_tensor.get(i_tnsr).data() +
                      volume_it.volume_offset()) ==
                    sliced_tensor.get(i_tnsr)[slice_it.volume_offset()]);
            }
            for (size_t i_tnsr = 0; i_tnsr < volume_tensor_IJ.size();
                 ++i_tnsr) {
              CHECK(*(volume_tensor_IJ[i_tnsr].data() +
                      volume_it.volume_offset()) ==
                    sliced_tensor_IJ[i_tnsr][slice_it.volume_offset()]);
            }
          }
        }
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SliceTensor", "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
