// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ContainerHelpers.hpp"

/// [get_element_example_indexing_callable]
struct ArrayOfArraysIndexFunctor {
  template <size_t OuterArraySize, size_t InnerArraySize, typename T>
  T& operator()(std::array<std::array<T, InnerArraySize>, OuterArraySize>&
                    array_of_arrays,
                size_t index) noexcept {
    return array_of_arrays.at(index % OuterArraySize)
        .at(index / OuterArraySize);
  }
};
/// [get_element_example_indexing_callable]

/// [get_size_example_size_callable]
struct ArrayOfArraysSizeFunctor {
  template <size_t OuterArraySize, size_t InnerArraySize, typename T>
  size_t operator()(
      const std::array<std::array<T, InnerArraySize>, OuterArraySize>&
      /*array_of_arrays*/) noexcept {
    return OuterArraySize * InnerArraySize;
  }
};
/// [get_size_example_size_callable]

namespace {
struct HalfDataVectorSize {
  size_t operator()(const DataVector& x) noexcept { return x.size() / 2; }
};

struct ReverseIndexStlVector {
  template <typename T>
  decltype(auto) operator()(T& x, size_t i) noexcept {
    return x.at(x.size() - i - 1);
  }
};

}  // namespace
SPECTRE_TEST_CASE("Unit.Utilities.ContainerHelpers", "[Unit][Utilities]") {
  const std::vector<double> constant_stl_vector{0.0, 1.0, 2.0, 3.0, 4.0,
                                                5.0, 6.0, 7.0, 8.0, 9.0};
  const double constant_fundamental = 10.0;
  const std::complex<double> constant_complex_of_fundamental =
      std::complex<double>(1.0, 5.0);
  const DataVector constant_spectre_vector(10, 2.0);
  std::vector<double> stl_vector{0.0, 1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0, 9.0};
  double fundamental = 10.0;
  std::complex<double> complex_of_fundamental = std::complex<double>(1.0, 5.0);
  std::array<std::array<double, 5>, 2> array_of_arrays = {
      {{{0.0, 2.0, 4.0, 6.0, 8.0}}, {{1.0, 3.0, 5.0, 7.0, 9.0}}}};

  DataVector spectre_vector(10, 2.0);
  REQUIRE(get_size(constant_stl_vector) == 10);
  REQUIRE(get_size(constant_fundamental) == 1);
  REQUIRE(get_size(constant_complex_of_fundamental) == 1);
  REQUIRE(get_size(constant_spectre_vector) == 10);
  REQUIRE(get_size(stl_vector) == 10);
  REQUIRE(get_size(fundamental) == 1);
  REQUIRE(get_size(complex_of_fundamental) == 1);
  REQUIRE(get_size(spectre_vector) == 10);
  REQUIRE(get_size(array_of_arrays, ArrayOfArraysSizeFunctor{}) == 10);

  CHECK(get_size(constant_spectre_vector, HalfDataVectorSize{}) == 5);
  CHECK(get_size(spectre_vector, HalfDataVectorSize{}) == 5);
  CHECK(get_size(constant_fundamental, HalfDataVectorSize{}) == 1);
  CHECK(get_size(fundamental, HalfDataVectorSize{}) == 1);

  std::vector<double> stl_vector_to_reverse{0.0, 1.0, 2.0, 3.0, 4.0,
                                            5.0, 6.0, 7.0, 8.0, 9.0};

  for (size_t i = 0; i < constant_stl_vector.size(); ++i) {
    get_element(stl_vector, i) = get_element(constant_stl_vector, i) * 2.0;
    get_element(stl_vector_to_reverse, i, ReverseIndexStlVector{}) =
        get_element(constant_stl_vector, i) * 2.0;
    get_element(fundamental, i) = get_element(constant_fundamental, i) * 2.0;
    get_element(complex_of_fundamental, i) =
        get_element(constant_complex_of_fundamental, i) * 2.0;
    get_element(spectre_vector, i) =
        get_element(constant_spectre_vector, i) * 2.0;
  }

  for (size_t i = 0; i < constant_stl_vector.size(); ++i) {
    CHECK(get_element(constant_stl_vector, i) == static_cast<double>(i));
    CHECK(get_element(constant_stl_vector, i, ReverseIndexStlVector{}) ==
          (9.0 - static_cast<double>(i)));
    CHECK(get_element(constant_fundamental, i) == 10.0);
    CHECK(get_element(constant_fundamental, i, ReverseIndexStlVector{}) ==
          10.0);
    CHECK(get_element(constant_complex_of_fundamental, i) ==
          std::complex<double>(1.0, 5.0));
    CHECK(get_element(constant_spectre_vector, i) == 2.0);
    CAPTURE(stl_vector);
    CAPTURE(stl_vector_to_reverse);
    CHECK(get_element(stl_vector, i) == 2.0 * static_cast<double>(i));
    CHECK(get_element(stl_vector, i, ReverseIndexStlVector{}) ==
          (9.0 - static_cast<double>(i)) * 2.0);
    CHECK(get_element(stl_vector_to_reverse, i) ==
          (9.0 - static_cast<double>(i)) * 2.0);
    CHECK(get_element(fundamental, i) == 20.0);
    CHECK(get_element(fundamental, i, ReverseIndexStlVector{}) == 20.0);
    CHECK(get_element(complex_of_fundamental, i) ==
          std::complex<double>(2.0, 10.0));
    CHECK(get_element(spectre_vector, i) == 4.0);
    CHECK(get_element(array_of_arrays, i, ArrayOfArraysIndexFunctor{}) ==
          static_cast<double>(i));
  }
}
