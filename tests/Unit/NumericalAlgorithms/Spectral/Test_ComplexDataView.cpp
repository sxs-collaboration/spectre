// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Spectral {
namespace Swsh {
namespace detail {
namespace {

template <ComplexRepresentation Representation>
void test_basic_view_functionality(
    const ComplexDataView<Representation>& view,
    std::complex<double>* const source_vec_data, const size_t view_size,
    const size_t offset) noexcept {
  CHECK(view.size() == view_size);
  // clang-tidy: This class uses manual memory management deliberately,
  // so silence complaints about pointer math and casts.
  // The reinterpret casts are intentional. See
  // NumericalAlgorithms/Spectral/ComplexDataView.cpp for an explanation
  if (Representation == ComplexRepresentation::Interleaved) {
    CHECK(view.real_data() ==
          reinterpret_cast<double*>(source_vec_data + offset));  // NOLINT
    CHECK(view.real_data() ==
          reinterpret_cast<double*>(source_vec_data + offset));  // NOLINT
  }
  CHECK(view.stride() ==
        (Representation == ComplexRepresentation::Interleaved ? 2 : 1));
}

template <ComplexRepresentation Representation>
void test_view() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-100.0, 100.0};
  UniformCustomDistribution<size_t> sdist{5, 50};
  const size_t overall_size = sdist(gen);
  const size_t view_size = sdist(gen) % (overall_size) + 1;
  const size_t offset = sdist(gen) % (overall_size - view_size + 1);
  const size_t stride = ComplexDataView<Representation>::stride();
  ComplexDataVector source_vec{overall_size};
  std::vector<std::complex<double>> ptr_source_vec(overall_size);
  ComplexDataVector assign_vec_1{view_size};
  ComplexDataVector assign_vec_2{view_size};
  ComplexDataVector assign_view_source{view_size};

  fill_with_random_values(make_not_null(&source_vec), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&ptr_source_vec), make_not_null(&gen),
                          make_not_null(&dist));

  ComplexDataView<Representation> vector_view{make_not_null(&source_vec),
                                              view_size, offset};
  test_basic_view_functionality(vector_view, source_vec.data(), view_size,
                                offset);

  const ComplexDataVector source_vec_copy = source_vec;
  // check conjugation utility
  vector_view.conjugate();
  for (size_t i = 0; i < view_size; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.real_data()[stride * i] ==
          real(source_vec_copy[i + offset]));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.imag_data()[stride * i] ==
          -imag(source_vec_copy[i + offset]));
  }

  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(source_vec[i + offset] == conj(source_vec_copy)[i + offset]);
  }

  // reverse the conjugation and check again
  vector_view.conjugate();
  for (size_t i = 0; i < view_size; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.real_data()[stride * i] ==
          real(source_vec_copy[i + offset]));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.imag_data()[stride * i] ==
          imag(source_vec_copy[i + offset]));
  }

  vector_view.copy_back_to_source();
  const ComplexDataView<Representation> ptr_view{
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      ptr_source_vec.data() + offset, view_size};
  test_basic_view_functionality(ptr_view, ptr_source_vec.data(), view_size,
                                offset);

  const ComplexDataVector pre_change =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      ComplexDataVector{source_vec.data() + offset, view_size};
  // check the various assignment operators
  for (size_t i = 0; i < view_size; i++) {
    assign_vec_1[i] = source_vec[i + offset] + std::complex<double>{1.0, 1.0};
    assign_vec_2[i] = source_vec[i + offset] + std::complex<double>{2.0, 2.0};
  }

  // real assignment
  vector_view.assign_real(real(assign_vec_1));
  for (size_t i = 0; i < view_size; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.real_data()[stride * i] == real(assign_vec_1)[i]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.imag_data()[stride * i] == imag(pre_change)[i]);
  }

  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(real(assign_vec_1[i]) == real(source_vec[i + offset]));
    CHECK(imag(pre_change[i]) == imag(source_vec[i + offset]));
  }

  // imag assignment
  vector_view.assign_imag(imag(assign_vec_2));
  for (size_t i = 0; i < view_size; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.real_data()[stride * i] == real(assign_vec_1)[i]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    CHECK(vector_view.imag_data()[stride * i] == imag(assign_vec_2)[i]);
  }

  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(real(assign_vec_1[i]) == real(source_vec[i + offset]));
    CHECK(imag(assign_vec_2[i]) == imag(source_vec[i + offset]));
  }

  // check self-assignment from the referenced vector
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
  assign_vec_1 = assign_vec_1;
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
  for (size_t i = 0; i < view_size; i++) {
    CHECK(real(assign_vec_1[i]) == real(source_vec[i + offset]));
    CHECK(imag(assign_vec_1[i]) == imag(assign_vec_1[i]));
  }

  // full assignment from vector
  fill_with_random_values(make_not_null(&assign_vec_1), make_not_null(&gen),
                          make_not_null(&dist));
  vector_view = assign_vec_1;
  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(source_vec[i + offset] == assign_vec_1[i]);
  }

  // full assignment from a view
  fill_with_random_values(make_not_null(&assign_view_source),
                          make_not_null(&gen), make_not_null(&dist));
  ComplexDataView<Representation> assign_view{
      make_not_null(&assign_view_source), assign_view_source.size()};
  vector_view = assign_view;
  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(source_vec[i + offset] == assign_view_source[i]);
  }

  // check self-assignment from the view
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
  // clang-tidy and gcc ignore for allowing the intentional self-assignment
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
  vector_view = vector_view; // NOLINT
#pragma GCC diagnostic pop
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
  for (size_t i = 0; i < view_size; i++) {
    CHECK(source_vec[i + offset] == assign_view_source[i]);
  }
  vector_view.copy_back_to_source();
  for (size_t i = 0; i < view_size; i++) {
    CHECK(source_vec[i + offset] == assign_view_source[i]);
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.ComplexDataView",
                  "[Unit][NumericalAlgorithms]") {
    {
        INFO("Test Interleaved view");
        test_view<ComplexRepresentation::Interleaved>();
    }
    {
        INFO("Test RealsThenImags view");
        test_view<ComplexRepresentation::RealsThenImags>();
    }
}

// spot-test two assignment asserts - they all call the same size-checking
// function, so this should be sufficiently robust.

// [[OutputRegex, Assignment must be to the same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.ComplexDataView.ComplexSizeError",
    "[Unit][NumericalAlgorithms]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 50};
  const size_t overall_size = sdist(gen);
  const size_t view_size = sdist(gen) % (overall_size - 1) + 1;
  const size_t offset = sdist(gen) % (overall_size - view_size + 1);

  ComplexDataVector source_vec{overall_size};
  ComplexDataView<ComplexRepresentation::Interleaved> vector_view_1{
      make_not_null(&source_vec), view_size, offset};
  ComplexDataView<ComplexRepresentation::Interleaved> vector_view_2{
      make_not_null(&source_vec), view_size + 1, offset};

  // this line should fail the size assert
  vector_view_1 = vector_view_2;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Assignment must be to the same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Spectral.ComplexDataView.RealSizeError",
    "[Unit][NumericalAlgorithms]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 50};
  const size_t overall_size = sdist(gen);
  const size_t view_size = sdist(gen) % (overall_size - 1) + 1;
  const size_t offset = sdist(gen) % (overall_size - view_size + 1);

  ComplexDataVector source_vec{overall_size};
  ComplexDataView<ComplexRepresentation::RealsThenImags> vector_view_1{
      make_not_null(&source_vec), view_size, offset};

  // this line should fail the size assert
  vector_view_1.assign_real(real(source_vec));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace
}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
