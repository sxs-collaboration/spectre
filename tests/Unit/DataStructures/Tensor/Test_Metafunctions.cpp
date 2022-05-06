// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdint>
#include <type_traits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;
namespace Frame {
struct BlockLogical;
struct Grid;
struct Inertial;
}  // namespace Frame

// Test check_index_symmetry
static_assert(TensorMetafunctions::check_index_symmetry_v<tmpl::list<>>,
              "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1>, SpatialIndex<3, UpLo::Lo, Frame::Grid>>,
              "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1>, SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>,
              "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1>, SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 1>, SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                  SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 1>, SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                  SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 1>, SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                  SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 1>, SpatialIndex<3, UpLo::Up, Frame::Grid>,
                  SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                  SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
              "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<2, 1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<2, 1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<2, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<2, 1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Lo, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<2, 1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Distorted>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<2, 1, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpacetimeIndex<3, UpLo::Up, Frame::BlockLogical>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");

// Test prepend_spacetime_index and prepend_spatial_index
static_assert(std::is_same_v<tnsr::aB<double, 3, Frame::Grid>,
                             TensorMetafunctions::prepend_spacetime_index<
                                 tnsr::A<double, 3, Frame::Grid>, 3, UpLo::Lo,
                                 Frame::Grid>>,
              "Failed testing prepend_spacetime_index");
static_assert(std::is_same_v<tnsr::iJ<double, 3, Frame::Grid>,
                             TensorMetafunctions::prepend_spatial_index<
                                 tnsr::I<double, 3, Frame::Grid>, 3, UpLo::Lo,
                                 Frame::Grid>>,
              "Failed testing prepend_spatial_index");

// Test remove_first_index
static_assert(
    std::is_same_v<Scalar<double>, TensorMetafunctions::remove_first_index<
                                       tnsr::a<double, 3, Frame::Grid>>>,
    "Failed testing remove_first_index");
static_assert(std::is_same_v<tnsr::A<double, 3, Frame::Grid>,
                             TensorMetafunctions::remove_first_index<
                                 tnsr::aB<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");
static_assert(std::is_same_v<tnsr::ab<double, 3, Frame::Grid>,
                             TensorMetafunctions::remove_first_index<
                                 tnsr::abc<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");
static_assert(
    std::is_same_v<
        tnsr::ab<double, 3, Frame::Grid>,
        TensorMetafunctions::remove_first_index<Tensor<
            double, tmpl::integral_list<std::int32_t, 2, 2, 1>,
            index_list<Tensor_detail::TensorIndexType<3, UpLo::Lo, Frame::Grid,
                                                      IndexType::Spacetime>,
                       Tensor_detail::TensorIndexType<3, UpLo::Lo, Frame::Grid,
                                                      IndexType::Spacetime>,
                       Tensor_detail::TensorIndexType<3, UpLo::Lo, Frame::Grid,
                                                      IndexType::Spacetime>>>>>,
    "Failed testing remove_first_index");
static_assert(std::is_same_v<tnsr::aa<double, 3, Frame::Grid>,
                             TensorMetafunctions::remove_first_index<
                                 tnsr::abb<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");

// Test swap_type
static_assert(
    std::is_same_v<tnsr::ij<double, 3>, TensorMetafunctions::swap_type<
                                            double, tnsr::ij<DataVector, 3>>>,
    "Failed testing swap_type");

// Test any_index_in_frame_v
static_assert(TensorMetafunctions::any_index_in_frame_v<
                  tnsr::iJ<double, 3, Frame::Grid>, Frame::Grid>,
              "Failed testing any_index_in_frame_v where it should match");
static_assert(not TensorMetafunctions::any_index_in_frame_v<
                  tnsr::iJ<double, 3, Frame::Grid>, Frame::Inertial>,
              "Failed testing any_index_in_frame_v where it should not match");
static_assert(
    TensorMetafunctions::any_index_in_frame_v<
        Tensor<
            double, tmpl::integral_list<std::int32_t, 2, 1>,
            index_list<Tensor_detail::TensorIndexType<
                           3, UpLo::Lo, Frame::Inertial, IndexType::Spacetime>,
                       Tensor_detail::TensorIndexType<3, UpLo::Lo, Frame::Grid,
                                                      IndexType::Spacetime>>>,
        Frame::Inertial>,
    "Failed testing any_index_in_frame_v for a tensor with different frames "
    "for different indices");
