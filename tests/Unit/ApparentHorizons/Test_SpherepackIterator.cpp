// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ApparentHorizons.SpherepackIterator",
                  "[ApparentHorizons][Unit]") {
  const std::vector<size_t> test_l = {0, 1, 1, 2, 2, 2, 3, 3, 3, 4,
                                      4, 4, 1, 2, 2, 3, 3, 4, 4};
  const std::vector<size_t> test_m = {0, 0, 1, 0, 1, 2, 0, 1, 2, 0,
                                      1, 2, 1, 1, 2, 1, 2, 1, 2};
  const std::vector<size_t> test_index = {0,   15,  20,  30,  35, 40, 45,
                                          50,  55,  60,  65,  70, 95, 110,
                                          115, 125, 130, 140, 145};

  /// [spherepack_iterator_example]
  const size_t l_max = 4;
  const size_t m_max = 2;
  const size_t stride = 5;
  SpherepackIterator iter(l_max, m_max, stride);
  // Allocate space for a SPHEREPACK array
  std::vector<double> array(iter.spherepack_array_size() * stride);
  // Set each array element equal to l+m for real part
  // and l-m for imaginary part.
  size_t i = 0;
  for (iter.reset(); iter; ++iter, ++i) {
    if (iter.coefficient_array() == SpherepackIterator::CoefficientArray::a) {
      array[iter()] = iter.l() + iter.m();
    } else {
      array[iter()] = iter.l() - iter.m();
    }
    CHECK(iter.l() == test_l[i]);
    CHECK(iter.m() == test_m[i]);
    CHECK(iter() == test_index[i]);
  }
  /// [spherepack_iterator_example]
  CHECK(iter.l_max() == 4);
  CHECK(iter.m_max() == 2);
  CHECK(iter.n_th() == 5);
  CHECK(iter.n_ph() == 5);
  for (i = 0; i < test_index.size(); ++i) {
    auto j = test_index[i];
    if (i > 11) {  // For specific test_index chosen above.
      // imag part
      CHECK(array[j] == test_l[i] - test_m[i]);
    } else {
      // real part
      CHECK(array[j] == test_l[i] + test_m[i]);
    }
  }

  // Test set functions
  CHECK(iter.set(2, 1, SpherepackIterator::CoefficientArray::b)() == 110);
  // Test the set function for the case l>m_max+1
  CHECK(iter.set(4, 1, SpherepackIterator::CoefficientArray::a)() == 65);
  CHECK(iter.set(4, 1, SpherepackIterator::CoefficientArray::b)() == 140);
  CHECK(iter.reset()() == 0);
  CHECK(iter.set(2, 1)() == 35);
  CHECK(iter.set(2, -1)() == 110);

  // Test coefficient_arrya stream operator (assumes output of last 'set').
  CHECK(get_output(iter.coefficient_array()) == "b");

  // Test inequality
  const SpherepackIterator iter2(3, 2, 5);  // Different lmax,mmax
  const SpherepackIterator iter3(4, 2, 4);  // Different stride
  const SpherepackIterator iter4(4, 2, 5);  // Different current state
  CHECK(iter2 != iter);
  CHECK(iter != iter2);
  CHECK(iter != iter3);
  CHECK(iter3 != iter);
  CHECK(iter4 != iter);
  CHECK(iter != iter4);

  test_copy_semantics(iter);
  const auto iter_copy = iter;
  CHECK(iter_copy == iter);
  test_move_semantics(std::move(iter), iter_copy, 3_st, 2_st, 3_st);
}
