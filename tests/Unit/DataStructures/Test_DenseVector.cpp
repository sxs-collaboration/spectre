// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DenseVector.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include "DataStructures/DataVector.hpp"

namespace {
struct DenseVectorOption {
  static constexpr OptionString help = {"A vector"};
  using type = DenseVector<double>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DenseVector", "[DataStructures][Unit]") {
  // Since `DenseVector` is just a thin wrapper around `blaze::DynamicVector` we
  // test a few operations and refer to Blaze for more thorough tests.
  DenseVector<double> vec(size_t{3}, 1.);
  CHECK(vec.size() == 3);
  CHECK(vec[0] == 1.);
  CHECK(vec[1] == 1.);
  CHECK(vec[2] == 1.);
  CHECK_ITERABLE_APPROX((DenseVector<int>{2, 1, 0} + DenseVector<int>{1, 3, 1}),
                        (DenseVector<int>{3, 4, 1}));
  DenseVector<double> a{2., 1., 2.};
  DenseVector<double> b{3., 1., 1.};
  DenseVector<double> c{0.2, 0., 0.5};
  CHECK_ITERABLE_APPROX(2. * a, (DenseVector<double>{4., 2., 4.}));
  CHECK_ITERABLE_APPROX(a * b, (DenseVector<double>{6., 1., 2.}));
  CHECK_ITERABLE_APPROX(a + b, (DenseVector<double>{5., 2., 3.}));
  CHECK_ITERABLE_APPROX(length(a), 3.);
  CHECK_ITERABLE_APPROX(dot(a, a), 9.);
  CHECK_ITERABLE_APPROX(dot(a, b), 9.);
  CHECK_ITERABLE_APPROX(dot(a, c), 1.4);

  CHECK(make_with_value<DenseVector<double>>(vec, 2.) ==
        DenseVector<double>(size_t{3}, 2.));

  test_serialization(vec);
  test_copy_semantics(vec);
  auto vec_copy = vec;
  test_move_semantics(std::move(vec), vec_copy);

  Options<tmpl::list<DenseVectorOption>> opts("");
  opts.parse("DenseVectorOption: [1, 2, 3]");
  DenseVector<double> expected{{1., 2., 3.}};
  CHECK(opts.get<DenseVectorOption>() == expected);
}
