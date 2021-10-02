// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

namespace {
template <typename VectorType>
void test_variables_slice() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 10};

  const size_t x_extents = sdist(gen);
  const size_t y_extents = sdist(gen);
  const size_t z_extents = sdist(gen);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> vars{
      x_extents * y_extents * z_extents};
  const size_t tensor_size =
      TestHelpers::Tags::Vector<VectorType>::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);

  // Test data_on_slice function by using a predictable data set where each
  // entry is assigned a value equal to its index
  for (size_t s = 0; s < vars.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    vars.data()[s] = s;  // NOLINT
  }
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_x(y_extents * z_extents, 0.);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_y(x_extents * z_extents, 0.);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = sdist(gen) % x_extents;
  const size_t y_offset = sdist(gen) % y_extents;
  const size_t z_offset = sdist(gen) % z_extents;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }

  CHECK(data_on_slice(vars, extents, 0, x_offset) == expected_vars_sliced_in_x);
  CHECK(data_on_slice(vars, extents, 1, y_offset) == expected_vars_sliced_in_y);
  CHECK(data_on_slice(vars, extents, 2, z_offset) == expected_vars_sliced_in_z);
}

template <typename VectorType>
void test_variables_add_slice_to_data() {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<
      tt::get_fundamental_type_t<typename VectorType::value_type>>
      dist{-100.0, 100.0};

  // Test adding two slices on different 'axes' to a Variables
  std::array<VectorType, 3> orig_vals;
  std::fill(orig_vals.begin(), orig_vals.end(), VectorType{8});
  fill_with_random_values(make_not_null(&orig_vals), make_not_null(&gen),
                          make_not_null(&dist));

  std::array<VectorType, 3> slice0_vals;
  std::fill(slice0_vals.begin(), slice0_vals.end(), VectorType{4});
  fill_with_random_values(make_not_null(&slice0_vals), make_not_null(&gen),
                          make_not_null(&dist));

  std::array<VectorType, 3> slice1_vals;
  std::fill(slice1_vals.begin(), slice1_vals.end(), VectorType{2});
  fill_with_random_values(make_not_null(&slice1_vals), make_not_null(&gen),
                          make_not_null(&dist));

  using Tensor = typename TestHelpers::Tags::Vector<VectorType>::type;
  const Index<2> extents{{{4, 2}}};
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> vars(
      extents.product());
  get<TestHelpers::Tags::Vector<VectorType>>(vars) =
      Tensor{{{orig_vals[0], orig_vals[1], orig_vals[2]}}};
  {
    const auto slice_extents = extents.slice_away(0);
    Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> slice(
        slice_extents.product(), 0.);
    get<TestHelpers::Tags::Vector<VectorType>>(slice) =
        Tensor{{{slice1_vals[0], slice1_vals[1], slice1_vals[2]}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 0, 2);
  }

  {
    const auto slice_extents = extents.slice_away(1);
    Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> slice(
        slice_extents.product(), 0.);
    get<TestHelpers::Tags::Vector<VectorType>>(slice) =
        Tensor{{{slice0_vals[0], slice0_vals[1], slice0_vals[2]}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 1, 1);
  }

  {
    // Test using add_slice_to_data with prefixed Variables
    const auto slice_extents = extents.slice_away(0);
    Variables<tmpl::list<
        TestHelpers::Tags::Prefix0<TestHelpers::Tags::Vector<VectorType>>>>
        slice(slice_extents.product(), 0.);
    get<TestHelpers::Tags::Prefix0<TestHelpers::Tags::Vector<VectorType>>>(
        slice) = Tensor{{{slice1_vals[0], slice1_vals[1], slice1_vals[2]}}};
    add_slice_to_data(make_not_null(&vars), slice, extents, 0, 2);
  }

  // The slice0_vals should have been added twice to the second half of each of
  // the three vectors in the tensor. The slice1_vals should have been added to
  // entries 2 and 6 in each vector.
  // clang-format off
  const Tensor expected{
      {{{orig_vals[0].at(0),
         orig_vals[0].at(1),
         orig_vals[0].at(2) + 2. * slice1_vals[0].at(0),
         orig_vals[0].at(3),
         orig_vals[0].at(4) + slice0_vals[0].at(0),
         orig_vals[0].at(5) + slice0_vals[0].at(1),
         orig_vals[0].at(6) + slice0_vals[0].at(2) + 2. * slice1_vals[0].at(1),
         orig_vals[0].at(7) + slice0_vals[0].at(3)},
        {orig_vals[1].at(0),
         orig_vals[1].at(1),
         orig_vals[1].at(2) + 2. * slice1_vals[1].at(0),
         orig_vals[1].at(3),
         orig_vals[1].at(4) + slice0_vals[1].at(0),
         orig_vals[1].at(5) + slice0_vals[1].at(1),
         orig_vals[1].at(6) + slice0_vals[1].at(2) + 2. * slice1_vals[1].at(1),
         orig_vals[1].at(7) + slice0_vals[1].at(3)},
        {orig_vals[2].at(0),
         orig_vals[2].at(1),
         orig_vals[2].at(2) + 2. * slice1_vals[2].at(0),
         orig_vals[2].at(3),
         orig_vals[2].at(4) + slice0_vals[2].at(0),
         orig_vals[2].at(5) + slice0_vals[2].at(1),
         orig_vals[2].at(6) + slice0_vals[2].at(2) + 2. * slice1_vals[2].at(1),
         orig_vals[2].at(7) + slice0_vals[2].at(3)}}}};
  // clang-format on

  CHECK_ITERABLE_APPROX(expected,
                        get<TestHelpers::Tags::Vector<VectorType>>(vars));
}

SPECTRE_TEST_CASE("Unit.DataStructures.SliceVariables",
                  "[DataStructures][Unit]") {
  INFO("Test Variables slice utilities") {
    test_variables_slice<ComplexDataVector>();
    test_variables_slice<ComplexModalVector>();
    test_variables_slice<DataVector>();
    test_variables_slice<ModalVector>();
  }
  INFO("Test adding slice values to Variables") {
    test_variables_add_slice_to_data<ComplexDataVector>();
    test_variables_add_slice_to_data<ComplexModalVector>();
    test_variables_add_slice_to_data<DataVector>();
    test_variables_add_slice_to_data<ModalVector>();
  }
}

// [[OutputRegex, volume_vars has wrong number of grid points.
//  Expected 8, got 10]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.add_slice_to_data.BadSize.volume",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>> vars(10, 0.);
  const Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>> slice(2,
                                                                           0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// clang-format off
// [[OutputRegex, vars_on_slice has wrong number of grid points.
//  Expected 2, got 5]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.Variables.add_slice_to_data.BadSize.slice",
    "[DataStructures][Unit]") {
  // clang-format on
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>> vars(8, 0.);
  const Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>> slice(5,
                                                                           0.);
  add_slice_to_data(make_not_null(&vars), slice, Index<2>{{{4, 2}}}, 0, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace
