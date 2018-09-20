// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
// IWYU pragma: no_forward_declare Tensor

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

// Test remove_first_index
static_assert(
    cpp17::is_same_v<Scalar<double>, TensorMetafunctions::remove_first_index<
                                         tnsr::a<double, 3, Frame::Grid>>>,
    "Failed testing remove_first_index");
static_assert(cpp17::is_same_v<tnsr::A<double, 3, Frame::Grid>,
                               TensorMetafunctions::remove_first_index<
                                   tnsr::aB<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");
static_assert(cpp17::is_same_v<tnsr::ab<double, 3, Frame::Grid>,
                               TensorMetafunctions::remove_first_index<
                                   tnsr::abc<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");
static_assert(
    cpp17::is_same_v<
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
static_assert(cpp17::is_same_v<tnsr::aa<double, 3, Frame::Grid>,
                               TensorMetafunctions::remove_first_index<
                                   tnsr::abb<double, 3, Frame::Grid>>>,
              "Failed testing remove_first_index");

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
        SpacetimeIndex<3, UpLo::Up, Frame::Logical>>,
    "Failed testing check_index_symmetry");
static_assert(
    not TensorMetafunctions::check_index_symmetry_v<
        Symmetry<1, 2, 1>, SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
        SpatialIndex<3, UpLo::Up, Frame::Inertial>>,
    "Failed testing check_index_symmetry");

// Test swap_type
static_assert(
    cpp17::is_same_v<tnsr::ij<double, 3>, TensorMetafunctions::swap_type<
                                              double, tnsr::ij<DataVector, 3>>>,
    "Failed testing swap_type");

static_assert(not cpp17::is_constructible_v<
                  Tensor<double, Symmetry<>, tmpl::list<>>, tmpl::list<>>,
              "Tensor construction failed to be SFINAE friendly");
static_assert(
    cpp17::is_constructible_v<Tensor<double, Symmetry<>, tmpl::list<>>, double>,
    "Tensor construction failed to be SFINAE friendly");

namespace {
template <typename DataType, size_t SpatialDim, typename Fr = Frame::Inertial,
          IndexType Index = IndexType::Spacetime>
using abcd = Tensor<
    DataType, tmpl::integral_list<std::int32_t, 4, 3, 2, 1>,
    index_list<
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>,
        Tensor_detail::TensorIndexType<SpatialDim, UpLo::Lo, Fr, Index>>>;

// Test construction of high-rank, high-dim tensors
constexpr abcd<double, 7> check_construction{};
// Silence unused variable warning
static_assert(check_construction.rank() == 4, "Wrong structure");
}  // namespace

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

  {
    /// [index_dim]
    using T = Tensor<double, Symmetry<1, 2, 3>,
                     index_list<SpacetimeIndex<2, UpLo::Up, Frame::Inertial>,
                                SpatialIndex<1, UpLo::Up, Frame::Inertial>,
                                SpatialIndex<2, UpLo::Up, Frame::Inertial>>>;
    const T t{};
    CHECK(index_dim<0>(t) == 3);
    CHECK(index_dim<1>(t) == 1);
    CHECK(index_dim<2>(t) == 2);
    CHECK(T::index_dim(0) == 3);
    CHECK(T::index_dim(1) == 1);
    CHECK(T::index_dim(2) == 2);
    CHECK(T::index_dims() == std::array<size_t, 3>{{3, 1, 2}});
    /// [index_dim]
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
  double i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : symmetric_rank2_dim4) {
    p = i;
    ++i;
  }
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      CHECK(symmetric_rank2_dim4.get(j, k) == symmetric_rank2_dim4.get(k, j));
    }
  }
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
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : symmetric_rank3_dim4) {
    p = i;
    ++i;
  }
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      for (size_t l = 0; l < 3; ++l) {
        CHECK(symmetric_rank3_dim4.get(j, k, l) ==
              symmetric_rank3_dim4.get(j, l, k));
        if (l != j) {
          CHECK(symmetric_rank3_dim4.get(j, k, l) !=
                symmetric_rank3_dim4.get(l, k, j));
        }
      }
    }
  }
  Tensor<double, Symmetry<1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      symmetric_all_rank3_dim4{};
  CHECK(3 == symmetric_all_rank3_dim4.rank());
  CHECK(20 == symmetric_all_rank3_dim4.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : symmetric_all_rank3_dim4) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(symmetric_all_rank3_dim4.get(l, j, k) ==
              symmetric_all_rank3_dim4.get(l, k, j));
        CHECK(symmetric_all_rank3_dim4.get(l, j, k) ==
              symmetric_all_rank3_dim4.get(j, k, l));
        CHECK(symmetric_all_rank3_dim4.get(l, j, k) ==
              symmetric_all_rank3_dim4.get(j, l, k));
        CHECK(symmetric_all_rank3_dim4.get(l, j, k) ==
              symmetric_all_rank3_dim4.get(k, l, j));
        CHECK(symmetric_all_rank3_dim4.get(l, j, k) ==
              symmetric_all_rank3_dim4.get(k, j, l));
      }
    }
  }
  Tensor<double, Symmetry<1, 1, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      iii{};
  CHECK(3 == iii.rank());
  CHECK(10 == iii.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : iii) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 2; ++l) {
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(iii.get(l, j, k) == iii.get(l, k, j));
        CHECK(iii.get(l, j, k) == iii.get(j, k, l));
        CHECK(iii.get(l, j, k) == iii.get(j, l, k));
        CHECK(iii.get(l, j, k) == iii.get(k, l, j));
        CHECK(iii.get(l, j, k) == iii.get(k, j, l));
      }
    }
  }
  Tensor<double, Symmetry<3, 2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abcc{};
  CHECK(4 == abcc.rank());
  CHECK(160 == abcc.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abcc) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abcc.get(l, j, k, m) == abcc.get(l, j, m, k));
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 3, 1, 3>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abcb{};
  CHECK(4 == abcb.rank());
  CHECK(160 == abcb.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abcb) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abcb.get(l, j, k, m) == abcb.get(l, m, k, j));
          if (j != l) {
            CHECK(abcb.get(l, j, k, m) != abcb.get(j, l, k, m));
          }
          if (k != l) {
            CHECK(abcb.get(l, j, k, m) != abcb.get(k, j, l, m));
          }
          if (j != k) {
            CHECK(abcb.get(l, j, k, m) != abcb.get(l, k, j, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<3, 2, 1, 3>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abca{};
  CHECK(4 == abca.rank());
  CHECK(160 == abca.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abca) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abca.get(l, j, k, m) == abca.get(m, j, k, l));
          if (j != l) {
            CHECK(abca.get(l, j, k, m) != abca.get(j, l, k, m));
          }
          if (k != l) {
            CHECK(abca.get(l, j, k, m) != abca.get(k, j, l, m));
          }
          if (k != j) {
            CHECK(abca.get(l, j, k, m) != abca.get(l, k, j, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<3, 2, 3, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abac{};
  CHECK(4 == abac.rank());
  CHECK(160 == abac.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abac) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abac.get(l, j, k, m) == abac.get(k, j, l, m));
          if (l != j) {
            CHECK(abac.get(l, j, k, m) != abac.get(j, l, k, m));
          }
          if (l != m) {
            CHECK(abac.get(l, j, k, m) != abac.get(m, j, k, l));
          }
          if (j != m) {
            CHECK(abac.get(l, j, k, m) != abac.get(l, m, k, j));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<3, 3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      aabc{};
  CHECK(4 == aabc.rank());
  CHECK(160 == aabc.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : aabc) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(aabc.get(l, j, k, m) == aabc.get(j, l, k, m));
          if (l != k) {
            CHECK(aabc.get(l, j, k, m) != aabc.get(k, j, l, m));
          }
          if (l != m) {
            CHECK(aabc.get(l, j, k, m) != aabc.get(m, j, k, l));
          }
          if (m != k) {
            CHECK(aabc.get(l, j, k, m) != aabc.get(l, j, m, k));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      aabb{};
  CHECK(4 == aabb.rank());
  CHECK(100 == aabb.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : aabb) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(aabb.get(l, j, k, m) == aabb.get(j, l, k, m));
          CHECK(aabb.get(l, j, k, m) == aabb.get(j, l, m, k));
          CHECK(aabb.get(l, j, k, m) == aabb.get(l, j, m, k));
          if (j != k) {
            CHECK(aabb.get(l, j, k, m) != aabb.get(l, k, j, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 1, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abab{};
  CHECK(4 == abab.rank());
  CHECK(100 == abab.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abab) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abab.get(l, j, k, m) == abab.get(k, j, l, m));
          CHECK(abab.get(l, j, k, m) == abab.get(k, m, l, j));
          CHECK(abab.get(l, j, k, m) == abab.get(l, m, k, j));
          if (l != j) {
            CHECK(abab.get(l, j, k, m) != abab.get(j, l, k, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abba{};
  CHECK(4 == abba.rank());
  CHECK(100 == abba.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abba) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abba.get(l, j, k, m) == abba.get(l, k, j, m));
          CHECK(abba.get(l, j, k, m) == abba.get(m, k, j, l));
          CHECK(abba.get(l, j, k, m) == abba.get(m, j, k, l));
          if (l != j) {
            CHECK(abba.get(l, j, k, m) != abba.get(j, l, k, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 2, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      aaab{};
  CHECK(4 == aaab.rank());
  CHECK(80 == aaab.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : aaab) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(aaab.get(l, j, k, m) == aaab.get(l, k, j, m));
          CHECK(aaab.get(l, j, k, m) == aaab.get(j, l, k, m));
          CHECK(aaab.get(l, j, k, m) == aaab.get(j, k, l, m));
          CHECK(aaab.get(l, j, k, m) == aaab.get(k, l, j, m));
          CHECK(aaab.get(l, j, k, m) == aaab.get(k, j, l, m));
          if (l != m) {
            CHECK(aaab.get(l, j, k, m) != aaab.get(m, j, k, l));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      aaba{};
  CHECK(4 == aaba.rank());
  CHECK(80 == aaba.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : aaba) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(aaba.get(l, j, k, m) == aaba.get(l, m, k, j));
          CHECK(aaba.get(l, j, k, m) == aaba.get(j, l, k, m));
          CHECK(aaba.get(l, j, k, m) == aaba.get(j, m, k, l));
          CHECK(aaba.get(l, j, k, m) == aaba.get(m, l, k, j));
          CHECK(aaba.get(l, j, k, m) == aaba.get(m, j, k, l));
          if(l != k) {
            CHECK(aaba.get(l, j, k, m) != aaba.get(k, j, l, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abaa{};
  CHECK(4 == abaa.rank());
  CHECK(80 == abaa.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abaa) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abaa.get(l, j, k, m) == abaa.get(l, j, m, k));
          CHECK(abaa.get(l, j, k, m) == abaa.get(k, j, m, l));
          CHECK(abaa.get(l, j, k, m) == abaa.get(k, j, l, m));
          CHECK(abaa.get(l, j, k, m) == abaa.get(m, j, l, k));
          CHECK(abaa.get(l, j, k, m) == abaa.get(m, j, k, l));
          if (l != j) {
            CHECK(abaa.get(l, j, k, m) != abaa.get(j, l, k, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      abbb{};
  CHECK(4 == abbb.rank());
  CHECK(80 == abbb.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : abbb) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(abbb.get(l, j, k, m) == abbb.get(l, j, m, k));
          CHECK(abbb.get(l, j, k, m) == abbb.get(l, k, m, j));
          CHECK(abbb.get(l, j, k, m) == abbb.get(l, k, j, m));
          CHECK(abbb.get(l, j, k, m) == abbb.get(l, m, k, j));
          CHECK(abbb.get(l, j, k, m) == abbb.get(l, m, j, k));
          if (l != j) {
            CHECK(abbb.get(l, j, k, m) != abbb.get(j, l, k, m));
          }
        }
      }
    }
  }
  Tensor<double, Symmetry<1, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      aaaa{};
  CHECK(4 == aaaa.rank());
  CHECK(35 == aaaa.size());
  i = 1.948;  // not a value likely to be default-constructed to
  for (auto& p : aaaa) {
    p = i;
    ++i;
  }
  for (size_t l = 0; l < 3; ++l) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(l, j, m, k));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(l, k, m, j));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(l, k, j, m));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(l, m, k, j));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(l, m, j, k));

          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, k, l, m));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, k, m, l));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, m, k, l));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, m, l, k));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, l, m, k));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(j, l, k, m));

          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, m, j, l));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, m, l, j));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, j, m, l));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, j, l, m));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, l, j, m));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(k, l, m, j));

          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, l, j, k));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, l, k, j));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, j, l, k));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, j, k, l));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, k, l, j));
          CHECK(aaaa.get(l, j, k, m) == aaaa.get(m, k, j, l));
        }
      }
    }
  }

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
  Tensor<DataVector, Symmetry<1, 2, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      tensor_a(1_st);
  Tensor<DataVector, Symmetry<1, 2, 2>,
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
  Tensor<DataVector, Symmetry<1, 2, 2>,
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
  Tensor<DataVector, Symmetry<1, 2, 2>,
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
  Tensor<DataVector, Symmetry<1, 2, 2>,
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

  CHECK(get_output(Tensor<DataVector, Symmetry<1>,
                          index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>(
            10_st, 2.0)) == compare_out);

  compare_out =
      "T()=(2,2,2,2,2,2,2,2,2,2)";
  CHECK(get_output(Scalar<DataVector>(10_st, 2.0)) == compare_out);

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
  tnsr::Abb<DataVector, dim - 1, Frame::Grid> tensor(1_st);
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
  tensor[1000];
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
  tensor[1000];
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
  tensor.multiplicity(1000);
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
  tensor.get_tensor_index(1000);
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
  tnsr::I<DataVector, 3, Frame::Grid> tensor_data_vector{
      {{{1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}}}};
  Scalar<DataVector> scalar_data_vector{{{{1., 2., 3.}}}};
  /// [init_vector]
  CHECK(std::make_pair(std::vector<std::string>{"x", "y", "z"},
                       std::vector<DataVector>{
                           {1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}}) ==
        tensor_data_vector.get_vector_of_data());
  CHECK(std::make_pair(std::vector<std::string>{"Scalar"},
                       std::vector<DataVector>{{1., 2., 3.}}) ==
        scalar_data_vector.get_vector_of_data());

  tnsr::I<double, 3, Frame::Grid> tensor_double{{{1.0, 2.0, 3.0}}};
  CHECK(std::make_pair(std::vector<std::string>{"x", "y", "z"},
                       std::vector<double>{1.0, 2.0, 3.0}) ==
        tensor_double.get_vector_of_data());

  const Scalar<double> scalar{0.8};
  CHECK(std::make_pair(std::vector<std::string>{"Scalar"},
                       std::vector<double>{0.8}) ==
        scalar.get_vector_of_data());
}

/// [example_spectre_test_case]
SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Frames",
                  "[Unit][DataStructures]") {
  CHECK("Logical" == get_output(Frame::Logical{}));
  CHECK("Grid" == get_output(Frame::Grid{}));
  CHECK("Inertial" == get_output(Frame::Inertial{}));
  CHECK("Distorted" == get_output(Frame::Distorted{}));
  CHECK("NoFrame" == get_output(Frame::NoFrame{}));
}
/// [example_spectre_test_case]
