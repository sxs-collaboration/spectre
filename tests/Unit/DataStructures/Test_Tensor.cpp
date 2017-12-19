// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"

/// [change_up_lo]
using Index = SpatialIndex<3, UpLo::Lo, Frame::Grid>;
using UpIndex = change_index_up_lo<Index>;
static_assert(cpp17::is_same_v<UpIndex, SpatialIndex<3, UpLo::Up, Frame::Grid>>,
              "Failed testing change_index_up_lo");
/// [change_up_lo]

/// [is_frame_physical]
static_assert(not Frame::is_frame_physical_v<Frame::Logical>,
              "Failed testing Frame::is_frame_physical");
static_assert(not Frame::is_frame_physical_v<Frame::Distorted>,
              "Failed testing Frame::is_frame_physical");
static_assert(not Frame::is_frame_physical_v<Frame::Grid>,
              "Failed testing Frame::is_frame_physical");
static_assert(Frame::is_frame_physical_v<Frame::Inertial>,
              "Failed testing Frame::is_frame_physical");
/// [is_frame_physical]

// Test Symmetry metafunction
static_assert(
    cpp17::is_same_v<Symmetry<4, 1>, tmpl::integral_list<std::int32_t, 2, 1>>,
    "Failed testing Symmetry");
static_assert(
    cpp17::is_same_v<Symmetry<1, 4>, tmpl::integral_list<std::int32_t, 2, 1>>,
    "Failed testing Symmetry");
static_assert(
    cpp17::is_same_v<Symmetry<4>, tmpl::integral_list<std::int32_t, 1>>,
    "Failed testing Symmetry");

// Test prepend_spacetime_index and prepend_spatial_index
static_assert(cpp17::is_same_v<tnsr::aB<double, 3, Frame::Grid>,
                               TensorMetafunctions::prepend_spacetime_index<
                                   tnsr::A<double, 3, Frame::Grid>, 3, UpLo::Lo,
                                   Frame::Grid>>,
              "Failed testing prepend_spacetime_index");
static_assert(cpp17::is_same_v<tnsr::iJ<double, 3, Frame::Grid>,
                               TensorMetafunctions::prepend_spatial_index<
                                   tnsr::I<double, 3, Frame::Grid>, 3, UpLo::Lo,
                                   Frame::Grid>>,
              "Failed testing prepend_spatial_index");

// Test check_index_symmetry
static_assert(
    TensorMetafunctions::check_index_symmetry_v<typelist<>, typelist<>>,
    "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1>, typelist<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>,
    "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1>, typelist<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>,
    "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1>, typelist<SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(
    TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 1>, typelist<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 1>, typelist<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 1>, typelist<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 1>, typelist<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 1>, typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                                 SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
    "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<2, 1, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<2, 1, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<2, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<2, 1, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<2, 1, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Distorted>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<2, 1, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 2, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 2, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpacetimeIndex<2, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 2, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 2, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpacetimeIndex<3, UpLo::Up, Frame::Logical>>>,
              "Failed testing check_index_symmetry");
static_assert(not TensorMetafunctions::check_index_symmetry_v<
                  Symmetry<1, 2, 1>,
                  typelist<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                           SpatialIndex<3, UpLo::Up, Frame::Inertial>>>,
              "Failed testing check_index_symmetry");

static_assert(not cpp17::is_constructible_v<
                  Tensor<double, Symmetry<>, typelist<>>, typelist<>>,
              "Tensor construction failed to be SFINAE friendly");
static_assert(
    cpp17::is_constructible_v<Tensor<double, Symmetry<>, typelist<>>, double>,
    "Tensor construction failed to be SFINAE friendly");

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.ComponentNames",
                  "[DataStructures][Unit]") {
  /// [spatial_vector]
  tnsr::I<double, 3, Frame::Grid> spatial_vector3{};
  /// [spatial_vector]
  CHECK(spatial_vector3.component_name(std::array<size_t, 1>{{0}}) == "x");
  CHECK(spatial_vector3.component_name(std::array<size_t, 1>{{1}}) == "y");
  CHECK(spatial_vector3.component_name(std::array<size_t, 1>{{2}}) == "z");

  /// [spacetime_vector]
  tnsr::A<double, 3, Frame::Grid> spacetime_vector3{};
  /// [spacetime_vector]
  CHECK(spacetime_vector3.component_name(std::array<size_t, 1>{{0}}) == "t");
  CHECK(spacetime_vector3.component_name(std::array<size_t, 1>{{1}}) == "x");
  CHECK(spacetime_vector3.component_name(std::array<size_t, 1>{{2}}) == "y");
  CHECK(spacetime_vector3.component_name(std::array<size_t, 1>{{3}}) == "z");

  /// [scalar]
  Tensor<double> scalar{};
  /// [scalar]
  CHECK(scalar.component_name() == "Scalar");

  /// [rank_3_122]
  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<1, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<1, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<1, UpLo::Lo, Frame::Grid>>>
      tensor_1{};
  /// [rank_3_122]
  CHECK(tensor_1.component_name(std::array<size_t, 3>{{0, 0, 0}}) == "txx");
  CHECK(tensor_1.component_name(std::array<size_t, 3>{{1, 0, 0}}) == "xxx");

  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<2, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<2, UpLo::Lo, Frame::Grid>>>
      tensor_2{};

  CHECK(tensor_2.component_name(std::array<size_t, 3>{{0, 0, 0}}) == "txx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{1, 0, 0}}) == "xxx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{2, 0, 0}}) == "yxx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{0, 1, 0}}) == "tyx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{1, 1, 0}}) == "xyx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{2, 1, 0}}) == "yyx");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{0, 1, 1}}) == "tyy");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{1, 1, 1}}) == "xyy");
  CHECK(tensor_2.component_name(std::array<size_t, 3>{{2, 1, 1}}) == "yyy");

  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_3{};
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 0, 0}}) == "xxx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 0, 0}}) == "yxx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 0, 0}}) == "zxx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 1, 0}}) == "xyx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 1, 0}}) == "yyx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 1, 0}}) == "zyx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 2, 0}}) == "xzx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 2, 0}}) == "yzx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 2, 0}}) == "zzx");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 1, 1}}) == "xyy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 1, 1}}) == "yyy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 1, 1}}) == "zyy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 2, 1}}) == "xzy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 2, 1}}) == "yzy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 2, 1}}) == "zzy");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{0, 2, 2}}) == "xzz");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{1, 2, 2}}) == "yzz");
  CHECK(tensor_3.component_name(std::array<size_t, 3>{{2, 2, 2}}) == "zzz");

  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_4{};
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 0, 0}}) == "ttt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 0, 0}}) == "xtt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 0, 0}}) == "ytt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 0, 0}}) == "ztt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 1, 0}}) == "txt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 1, 0}}) == "xxt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 1, 0}}) == "yxt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 1, 0}}) == "zxt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 2, 0}}) == "tyt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 2, 0}}) == "xyt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 2, 0}}) == "yyt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 2, 0}}) == "zyt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 3, 0}}) == "tzt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 3, 0}}) == "xzt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 3, 0}}) == "yzt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 3, 0}}) == "zzt");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 1, 1}}) == "txx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 1, 1}}) == "xxx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 1, 1}}) == "yxx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 1, 1}}) == "zxx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 2, 1}}) == "tyx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 2, 1}}) == "xyx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 2, 1}}) == "yyx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 2, 1}}) == "zyx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 3, 1}}) == "tzx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 3, 1}}) == "xzx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 3, 1}}) == "yzx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 3, 1}}) == "zzx");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 2, 2}}) == "tyy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 2, 2}}) == "xyy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 2, 2}}) == "yyy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 2, 2}}) == "zyy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 3, 2}}) == "tzy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 3, 2}}) == "xzy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 3, 2}}) == "yzy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 3, 2}}) == "zzy");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{0, 3, 3}}) == "tzz");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{1, 3, 3}}) == "xzz");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{2, 3, 3}}) == "yzz");
  CHECK(tensor_4.component_name(std::array<size_t, 3>{{3, 3, 3}}) == "zzz");

  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_5{};
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 0, 0}},
                                make_array<3>(std::string("abcd"))) == "aaa");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 0, 0}},
                                make_array<3>(std::string("abcd"))) == "baa");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 0, 0}},
                                make_array<3>(std::string("abcd"))) == "caa");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 0, 0}},
                                make_array<3>(std::string("abcd"))) == "daa");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 1, 0}},
                                make_array<3>(std::string("abcd"))) == "aba");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 1, 0}},
                                make_array<3>(std::string("abcd"))) == "bba");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 1, 0}},
                                make_array<3>(std::string("abcd"))) == "cba");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 1, 0}},
                                make_array<3>(std::string("abcd"))) == "dba");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 2, 0}},
                                make_array<3>(std::string("abcd"))) == "aca");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 2, 0}},
                                make_array<3>(std::string("abcd"))) == "bca");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 2, 0}},
                                make_array<3>(std::string("abcd"))) == "cca");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 2, 0}},
                                make_array<3>(std::string("abcd"))) == "dca");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 3, 0}},
                                make_array<3>(std::string("abcd"))) == "ada");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 3, 0}},
                                make_array<3>(std::string("abcd"))) == "bda");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 3, 0}},
                                make_array<3>(std::string("abcd"))) == "cda");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 3, 0}},
                                make_array<3>(std::string("abcd"))) == "dda");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 1, 1}},
                                make_array<3>(std::string("abcd"))) == "abb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 1, 1}},
                                make_array<3>(std::string("abcd"))) == "bbb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 1, 1}},
                                make_array<3>(std::string("abcd"))) == "cbb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 1, 1}},
                                make_array<3>(std::string("abcd"))) == "dbb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 2, 1}},
                                make_array<3>(std::string("abcd"))) == "acb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 2, 1}},
                                make_array<3>(std::string("abcd"))) == "bcb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 2, 1}},
                                make_array<3>(std::string("abcd"))) == "ccb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 2, 1}},
                                make_array<3>(std::string("abcd"))) == "dcb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 3, 1}},
                                make_array<3>(std::string("abcd"))) == "adb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 3, 1}},
                                make_array<3>(std::string("abcd"))) == "bdb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 3, 1}},
                                make_array<3>(std::string("abcd"))) == "cdb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 3, 1}},
                                make_array<3>(std::string("abcd"))) == "ddb");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 2, 2}},
                                make_array<3>(std::string("abcd"))) == "acc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 2, 2}},
                                make_array<3>(std::string("abcd"))) == "bcc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 2, 2}},
                                make_array<3>(std::string("abcd"))) == "ccc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 2, 2}},
                                make_array<3>(std::string("abcd"))) == "dcc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 3, 2}},
                                make_array<3>(std::string("abcd"))) == "adc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 3, 2}},
                                make_array<3>(std::string("abcd"))) == "bdc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 3, 2}},
                                make_array<3>(std::string("abcd"))) == "cdc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 3, 2}},
                                make_array<3>(std::string("abcd"))) == "ddc");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{0, 3, 3}},
                                make_array<3>(std::string("abcd"))) == "add");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{1, 3, 3}},
                                make_array<3>(std::string("abcd"))) == "bdd");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{2, 3, 3}},
                                make_array<3>(std::string("abcd"))) == "cdd");
  CHECK(tensor_5.component_name(std::array<size_t, 3>{{3, 3, 3}},
                                make_array<3>(std::string("abcd"))) == "ddd");
}

// [[OutputRegex, Tensor dim\[0\] must be 1,2,3, or 4 for default axis_labels]]
SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.BadDim1",
                  "[DataStructures][Unit]") {
  ERROR_TEST();
  Tensor<double, Symmetry<1>,
         index_list<SpacetimeIndex<5, UpLo::Lo, Frame::Grid>>>
      tensor_5{3.0};
  std::stringstream os;
  os << tensor_5.component_name(std::array<size_t, 1>{{4}});
}

// [[OutputRegex, Tensor dim\[0\] must be 1,2, or 3 for default axis_labels]]
SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.BadDim2",
                  "[DataStructures][Unit]") {
  ERROR_TEST();
  Tensor<double, Symmetry<1>,
         index_list<SpatialIndex<6, UpLo::Lo, Frame::Grid>>>
      tensor_6{3.0};
  std::stringstream os;
  os << tensor_6.component_name(std::array<size_t, 1>{{4}});
}

// [[OutputRegex, Dimension mismatch: Tensor has dim = 4, but you specified 8
// different labels in abcdefgh]]
SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.StreamBad",
                  "[DataStructures][Unit]") {
  ERROR_TEST();
  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_5{};
  CHECK(tensor_5.component_name(
            tensor_5.get_tensor_index(size_t{0}),  // 0 can be a pointer
            make_array<3>(std::string("abcdefgh"))) == "aaa");
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.RankAndSize",
                  "[DataStructures][Unit]") {
  {
    Scalar<double> scalar{2.8};
    CHECK(scalar.multiplicity(0_st) == 1);  // 0 can be a pointer
    CHECK(scalar.symmetries() == (std::array<int, 0>{}));
    CHECK(scalar.get() == 2.8);
  }

  {
    Tensor<double, Symmetry<3>,
           index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
        spatial_vector3{std::array<double, 3>{{1, 8, 3}}};
    CHECK(index_dim<0>(spatial_vector3) == 3);
    CHECK(1 == spatial_vector3.rank());
    CHECK(3 == spatial_vector3.size());
    CHECK(spatial_vector3.get(0) == 1);
    CHECK(spatial_vector3.get(1) == 8);
    CHECK(spatial_vector3.get(2) == 3);
    CHECK(get<0>(spatial_vector3) == 1);
    CHECK(get<1>(spatial_vector3) == 8);
    CHECK(get<2>(spatial_vector3) == 3);
  }

  {
    // test const functions
    const Tensor<double, Symmetry<3>,
                 index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
        spatial_vector3{std::array<double, 3>{{1, 8, 3}}};
    CHECK(Scalar<double>{}.multiplicity(0_st) == 1);  // 0 can be a pointer
    CHECK(index_dim<0>(spatial_vector3) == 3);
    CHECK(1 == spatial_vector3.rank());
    CHECK(3 == spatial_vector3.size());
    CHECK(spatial_vector3.get(0) == 1);
    CHECK(spatial_vector3.get(1) == 8);
    CHECK(spatial_vector3.get(2) == 3);
    CHECK(get<0>(spatial_vector3) == 1);
    CHECK(get<1>(spatial_vector3) == 8);
    CHECK(get<2>(spatial_vector3) == 3);
  }

  Tensor<double, Symmetry<3>,
         index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      spatial_vector4{};
  CHECK(1 == spatial_vector4.rank());
  CHECK(4 == spatial_vector4.size());
  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      symmetric_rank2_dim4{};
  CHECK(2 == symmetric_rank2_dim4.rank());
  CHECK(10 == symmetric_rank2_dim4.size());
  Tensor<double, Symmetry<1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      rank2_4{};
  CHECK(2 == rank2_4.rank());
  CHECK(16 == rank2_4.size());
  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      symmetric_rank3_dim4{};
  CHECK(3 == symmetric_rank3_dim4.rank());
  CHECK(40 == symmetric_rank3_dim4.size());
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Indices",
                  "[DataStructures][Unit]") {
  // Tests that iterators correctly handle the symmetries. However, as a result
  // the test is implementation defined
  Tensor<double, Symmetry<1, 2, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor{};
  auto it = tensor.begin();
  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = j; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(make_array(k, i, j) == tensor.get_tensor_index(it));
        ++it;
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Iterating",
                  "[DataStructures][Unit]") {
  // Fills two tensors with values such that A+B =1 and then checks that this is
  // the case doing a for (int...) loop over the Tensor's underlying storage
  // vector, which is iterating over all independent components.
  const int dim = 4;
  Tensor<std::vector<double>, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_a(1_st);
  Tensor<std::vector<double>, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_b(1_st);

  CHECK(dim * dim * (dim + 1) / 2 == tensor_a.size());
  CHECK(tensor_a.size() == tensor_b.size());

  // Fill the tensors
  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      for (size_t k = j; k < dim; k++) {
        tensor_a.get(i, j, k)[0] = (i + 1.0) * (j + 2.0) * k;
        tensor_b.get(i, j, k)[0] = 1.0 - (i + 1.0) * (j + 2.0) * k;
      }
    }
  }

  // Test simple iteration with an integer
  size_t count = 0;
  for (size_t i = 0; i < tensor_a.size(); ++i) {
    CHECK(tensor_a[i][0] + tensor_b[i][0] == 1);
    count++;
  }
  CHECK(count == dim * dim * (dim + 1) / 2);

  // Check iteration with an integer and forward iterator
  count = 0;
  for (auto p = tensor_a.begin(); p != tensor_a.end(); ++p) {
    CHECK(tensor_a[count][0] == (*p)[0]);
    CHECK(&tensor_a[count] == &(*p));
    count++;
  }
  CHECK(count == tensor_a.size());
  // Check const version
  count = 0;
  for (const auto& p : tensor_a) {
    CHECK(tensor_a[count][0] == p[0]);
    CHECK(&tensor_a[count] == &p);
    count++;
  }
  CHECK(count == tensor_a.size());
  // Check iteration with an integer and reverse iterator
  count = tensor_a.size();
  for (auto p = tensor_a.rbegin(); p != tensor_a.rend(); ++p) {
    CHECK(tensor_a[count - 1][0] == (*p)[0]);
    CHECK(&tensor_a[count - 1] == &(*p));
    count--;
  }
  CHECK(count == 0);

  // Check sum of components
  const int dim2 = 3;
  Tensor<std::vector<double>, Symmetry<1, 2, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_c(1_st);
  for (size_t k = 0; k < dim2; k++) {
    for (size_t i = 0; i < dim2; i++) {
      for (size_t j = i; j < dim2; j++) {
        tensor_c.get(k, i, j)[0] = 100 * (k + 1) + 10 * (i + 1) + j + 1;
      }
    }
  }

  count = 0;
  int sum = 0;
  for (const auto& elem : tensor_c) {
    ++count;
    sum += elem[0];
  }
  CHECK(count == 18);
  CHECK(sum == 3942);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.IndexByVector",
                  "[DataStructures][Unit]") {
  Tensor<std::vector<double>, Symmetry<1, 2, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor(1_st);
  // These numbers are deliberately chosen to be obscure for debugging
  // purposes.
  double inc = 137.4;
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    (*it)[0] = inc;
    // A.get_tensor_index(it) returns a vector<int> which is used to test
    // A(vector<int>)
    CHECK(tensor.get(tensor.get_tensor_index(it))[0] == inc);
    inc += 23.1;
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Multiplicity",
                  "[DataStructures][Unit]") {
  Tensor<std::vector<double>, Symmetry<1, 2, 2>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor;

  // Test multiplicity by iterator
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    const auto& indices = tensor.get_tensor_index(it);
    const size_t multiplicity = (indices[1] == indices[2] ? 1 : 2);
    CHECK(multiplicity == tensor.multiplicity(it));
  }

  // Test multiplicity by integer
  for (size_t i = 0; i < tensor.size(); ++i) {
    const auto& indices = tensor.get_tensor_index(i);
    const size_t multiplicity = (indices[1] == indices[2] ? 1 : 2);
    CHECK(multiplicity == tensor.multiplicity(i));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.StreamData",
                  "[DataStructures][Unit]") {
  std::string compare_out =
      "T(0)=(2,2,2,2,2,2,2,2,2,2)\n"
      "T(1)=(2,2,2,2,2,2,2,2,2,2)\n"
      "T(2)=(2,2,2,2,2,2,2,2,2,2)";

  CHECK(get_output(Tensor<std::vector<double>, Symmetry<1>,
                          index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>(
            10_st, 2.0)) == compare_out);

  compare_out =
      "T()=(2,2,2,2,2,2,2,2,2,2)";
  CHECK(get_output(Scalar<std::vector<double>>(10_st, 2.0)) == compare_out);

  compare_out =
      "T(0,0,0)=0\n"
      "T(1,0,0)=1\n"
      "T(2,0,0)=2\n"
      "T(0,1,0)=3\n"
      "T(1,1,0)=4\n"
      "T(2,1,0)=5\n"
      "T(0,2,0)=6\n"
      "T(1,2,0)=7\n"
      "T(2,2,0)=8\n"
      "T(0,1,1)=9\n"
      "T(1,1,1)=10\n"
      "T(2,1,1)=11\n"
      "T(0,2,1)=12\n"
      "T(1,2,1)=13\n"
      "T(2,2,1)=14\n"
      "T(0,2,2)=15\n"
      "T(1,2,2)=16\n"
      "T(2,2,2)=17";

  CHECK(get_output([]() {
          tnsr::abb<double, 2, Frame::Inertial> tensor{};
          std::iota(tensor.begin(), tensor.end(), 0);
          return tensor;
        }()) == compare_out);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.StreamStructure",
                  "[DataStructures][Unit]") {
  {
    const Scalar<double> tensor{};
    CHECK(get_output(tensor.symmetries()) == "()");
    CHECK(get_output(tensor.index_types()) == "()");
    CHECK(get_output(tensor.index_dims()) == "()");
    CHECK(get_output(tensor.index_valences()) == "()");
    CHECK(get_output(tensor.index_frames()) == "()");
  }
  {
    using structure = Tensor_detail::Structure<Symmetry<1, 1, 3, 2>,
                             SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                             SpatialIndex<2, UpLo::Lo, Frame::Inertial>,
                             SpacetimeIndex<3, UpLo::Lo, Frame::Logical>,
                             SpacetimeIndex<2, UpLo::Up, Frame::Distorted>>;

    CHECK(get_output(structure::symmetries()) == "(3,3,2,1)");
    CHECK(get_output(structure::index_types()) ==
          "(Spatial,Spatial,Spacetime,Spacetime)");
    CHECK(get_output(structure::dims()) == "(2,2,4,3)");
    CHECK(get_output(structure::index_valences()) == "(Lo,Lo,Lo,Up)");
    CHECK(get_output(structure::index_frames()) ==
          "(Inertial,Inertial,Logical,Distorted)");
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Structure.Indices",
                  "[DataStructures][Unit]") {
  const int dim = 3;
  Tensor_detail::Structure<Symmetry<2, 1, 1>,
                           SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                           SpatialIndex<dim, UpLo::Lo, Frame::Grid>,
                           SpatialIndex<dim, UpLo::Lo, Frame::Grid>>
      tensor;
  for (size_t j = 0; j < dim; ++j) {
    for (size_t k = j; k < dim; ++k) {
      for (size_t i = 0; i < dim; ++i) {
        CHECK(tensor.get_storage_index(std::array<size_t, 3>{{i, j, k}}) ==
              tensor.get_storage_index(tensor.get_canonical_tensor_index(
                  tensor.get_storage_index(std::array<size_t, 3>{{i, j, k}}))));
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Serialization.Tensor",
                  "[DataStructures][Unit][Serialization]") {
  constexpr size_t dim = 4;
  tnsr::Abb<std::vector<double>, dim - 1, Frame::Grid> tensor(1_st);
  // Fill the tensors
  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      for (size_t k = j; k < dim; k++) {
        tensor.get(i, j, k)[0] = (i + 1.0) * (j + 2.0) * k;
      }
    }
  }
  test_serialization(tensor);
}

// [[OutputRegex, Expects violated: index >= 0 and index < narrow_cast<Size>]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.out_of_bounds_subscript",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  tnsr::Abb<double, 3, Frame::Grid> tensor(1_st);
  auto& t = tensor[1000];
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Expects violated: index >= 0 and index < narrow_cast<Size>]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.const_out_of_bounds_subscript",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const tnsr::Abb<double, 3, Frame::Grid> tensor(1_st);
  const auto& t = tensor[1000];
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Expects violated: index >= 0 and index < narrow_cast<Size>]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.const_out_of_bounds_multiplicity",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Scalar<double> tensor(1_st);
  const auto& t = tensor.multiplicity(1000);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Expects violated: index >= 0 and index < narrow_cast<Size>]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Tensor.const_out_of_bounds_get_tensor_index_vector",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  tnsr::I<double, 3, Frame::Grid> tensor(1_st);
  const auto& t = tensor.get_tensor_index(1000);
  ERROR("Bad test end");
#endif
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.IndexType",
                  "[Unit][DataStructures]") {
  CHECK(get_output(Frame::Logical{}) == "Logical");
  CHECK(get_output(Frame::Grid{}) == "Grid");
  CHECK(get_output(Frame::Distorted{}) == "Distorted");
  CHECK(get_output(Frame::Inertial{}) == "Inertial");
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.GetVectorOfData",
                  "[Unit][DataStructures]") {
  // NOTE: This test depends on the implementation of serialize and Tensor,
  // but that is inevitable without making the test more complicated.
  /// [init_vector]
  tnsr::I<std::vector<double>, 3, Frame::Grid> tensor_std_vector(
      {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}});
  /// [init_vector]
  CHECK(std::make_pair(std::vector<std::string>{"x", "y", "z"},
                       std::vector<std::vector<double>>{
                           {1, 2, 3}, {4, 5, 6}, {7, 8, 9}}) ==
        tensor_std_vector.get_vector_of_data());

  tnsr::I<double, 3, Frame::Grid> tensor_double{{{1.0, 2.0, 3.0}}};
  CHECK(std::make_pair(std::vector<std::string>{"x", "y", "z"},
                       std::vector<double>{1.0, 2.0, 3.0}) ==
        tensor_double.get_vector_of_data());

  const Scalar<double> scalar{0.8};
  CHECK(std::make_pair(std::vector<std::string>{"Scalar"},
                       std::vector<double>{0.8}) ==
        scalar.get_vector_of_data());
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Frames",
                  "[Unit][DataStructures]") {
  CHECK("Logical" == get_output(Frame::Logical{}));
  CHECK("Grid" == get_output(Frame::Grid{}));
  CHECK("Inertial" == get_output(Frame::Inertial{}));
  CHECK("Distorted" == get_output(Frame::Distorted{}));
  CHECK("NoFrame" == get_output(Frame::NoFrame{}));
}
