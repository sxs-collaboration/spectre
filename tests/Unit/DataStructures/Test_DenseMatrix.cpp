// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>

#include "DataStructures/DenseMatrix.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include "DataStructures/DataVector.hpp"

namespace {
struct RowMajorMatrix {
  static constexpr OptionString help = {"A row-major matrix"};
  using type = DenseMatrix<double, blaze::rowMajor>;
};
struct ColumnMajorMatrix {
  static constexpr OptionString help = {"A column-major matrix"};
  using type = DenseMatrix<double, blaze::columnMajor>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DenseMatrix", "[DataStructures][Unit]") {
  // Since `DenseMatrix` is just a thin wrapper around `blaze::DynamicMatrix` we
  // test a few operations and refer to Blaze for more thorough tests.
  DenseMatrix<int> A{{2, 1}, {0, 1}};
  CHECK(A.rows() == 2);
  CHECK(A.columns() == 2);
  CHECK(A(0, 0) == 2);
  CHECK(A(0, 1) == 1);
  CHECK(A(1, 0) == 0);
  CHECK(A(1, 1) == 1);
  DenseMatrix<int> B{{1, 0}, {3, 2}};
  CHECK(A + B == DenseMatrix<int>{{3, 1}, {3, 3}});
  CHECK(2 * A == DenseMatrix<int>{{4, 2}, {0, 2}});
  CHECK(A * B == DenseMatrix<int>{{5, 2}, {3, 2}});
  CHECK(B * A == DenseMatrix<int>{{2, 1}, {6, 5}});

  const DenseVector<int> a{1, 3};
  CHECK(A * a == DenseVector<int>{5, 3});
  CHECK(trans(a) * A == DenseVector<int>{2, 4});

  Matrix matrix{{1., 0., 2.}, {3., 1.5, 0.5}};
  CHECK(matrix.rows() == 2);
  CHECK(matrix.columns() == 3);
  CHECK(matrix(0, 0) == 1.);
  CHECK(matrix(0, 1) == 0.);
  CHECK(matrix(0, 2) == 2.);
  CHECK(matrix(1, 0) == 3.);
  CHECK(matrix(1, 1) == 1.5);
  CHECK(matrix(1, 2) == 0.5);

  test_serialization(A);
  test_copy_semantics(A);
  auto matrix_copy = A;
  test_move_semantics(std::move(A), matrix_copy);

  {
    Options<tmpl::list<RowMajorMatrix, ColumnMajorMatrix>> opts("");
    opts.parse("RowMajorMatrix: [[1, 2], [3, 4]]");
    opts.parse("ColumnMajorMatrix: [[1, 2], [3, 4]]");
    DenseMatrix<double> expected{{1., 2.}, {3., 4.}};
    CHECK(opts.get<RowMajorMatrix>() == expected);
    CHECK(opts.get<ColumnMajorMatrix>() == expected);
  }
  {
    Options<tmpl::list<RowMajorMatrix>> opts("");
    opts.parse("RowMajorMatrix: []");
    CHECK(opts.get<RowMajorMatrix>() == DenseMatrix<double>(0, 0));
  }
  {
    Options<tmpl::list<RowMajorMatrix>> opts("");
    opts.parse("RowMajorMatrix: [[], [], []]");
    CHECK(opts.get<RowMajorMatrix>() == DenseMatrix<double>(3, 0));
  }
}

// [[OutputRegex, All matrix columns must have the same size.]]
SPECTRE_TEST_CASE("Unit.DataStructures.DenseMatrix.InvalidOptions",
                  "[DataStructures][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<RowMajorMatrix>> opts("");
  opts.parse("RowMajorMatrix: [[1], [1, 2], [1, 2]]");
  opts.get<RowMajorMatrix>();
}
