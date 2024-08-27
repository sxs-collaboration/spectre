// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/CompressedMatrix.hpp"
#include "DataStructures/CompressedVector.hpp"
#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/StaticMatrix.hpp"
#include "DataStructures/StaticVector.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.BlazeInteroperability",
                  "[DataStructures][Unit]") {
  {
    INFO("StaticVector");
    test_serialization(blaze::StaticVector<double, 4>{0., 1., 0., 2.});
    CHECK(make_with_value<blaze::StaticVector<double, 3>>(
              blaze::DynamicVector<double>(3), 1.) ==
          blaze::StaticVector<double, 3>(1.));
    CHECK(TestHelpers::test_creation<blaze::StaticVector<double, 4>>(
              "[0., 1., 0., 2.]") ==
          blaze::StaticVector<double, 4>{0., 1., 0., 2.});

    blaze::StaticVector<double, 4> v{0., 1., 0., 2.};
    set_number_of_grid_points(make_not_null(&v), 4_st);
    CHECK(v == blaze::StaticVector<double, 4>{0., 1., 0., 2.});
#ifdef SPECTRE_DEBUG
    CHECK_THROWS_WITH(set_number_of_grid_points(make_not_null(&v), 3_st),
                      Catch::Matchers::ContainsSubstring(
                          "Tried to resize a StaticVector to 3"));
#endif  // SPECTRE_DEBUG
  }
  {
    INFO("DynamicVector");
    test_serialization(blaze::DynamicVector<double>{0., 1., 0., 2.});
    CHECK(make_with_value<blaze::DynamicVector<double>>(
              blaze::DynamicVector<double>(3), 1.) ==
          blaze::DynamicVector<double>(3, 1.));
    CHECK(TestHelpers::test_creation<blaze::DynamicVector<double>>(
              "[0., 1., 0., 2.]") ==
          blaze::DynamicVector<double>{0., 1., 0., 2.});

    blaze::DynamicVector<double> v{0., 1., 0., 2.};
    set_number_of_grid_points(make_not_null(&v), 4_st);
    CHECK(v == blaze::DynamicVector<double>{0., 1., 0., 2.});
    set_number_of_grid_points(make_not_null(&v), 3_st);
    CHECK(v.size() == 3);
  }
  {
    INFO("Complex DynamicVector");
    using namespace std::complex_literals;
    test_serialization(blaze::DynamicVector<std::complex<double>>{
        0. + 1.i, 1. + 2.i, 0. - 1.i, 2. + 3.i});
    CHECK(make_with_value<blaze::DynamicVector<std::complex<double>>>(
              blaze::DynamicVector<std::complex<double>>(3), 1.) ==
          blaze::DynamicVector<std::complex<double>>(3, 1.));
    CHECK(
        TestHelpers::test_creation<blaze::DynamicVector<std::complex<double>>>(
            "[[0., 1.], [1., 2.], [0., -1.], [2., 3.]]") ==
        blaze::DynamicVector<std::complex<double>>{0. + 1.i, 1. + 2.i, 0. - 1.i,
                                                   2. + 3.i});

    blaze::DynamicVector<std::complex<double>> v{0. + 1.i, 1. + 2.i, 0. - 1.i,
                                                 2. + 3.i};
    set_number_of_grid_points(make_not_null(&v), 4_st);
    CHECK(v == blaze::DynamicVector<std::complex<double>>{0. + 1.i, 1. + 2.i,
                                                          0. - 1.i, 2. + 3.i});
    set_number_of_grid_points(make_not_null(&v), 3_st);
    CHECK(v.size() == 3);
  }
  {
    INFO("CompressedVector");
    test_serialization(blaze::CompressedVector<double>{0., 1., 0., 2.});
    CHECK(serialize_and_deserialize(
              blaze::CompressedVector<double>{0., 1., 0., 2.})
              .nonZeros() == 2);
    CHECK(TestHelpers::test_creation<blaze::CompressedVector<double>>(
              "[0., 1., 0., 2.]") ==
          blaze::CompressedVector<double>{0., 1., 0., 2.});
    CHECK(TestHelpers::test_creation<blaze::CompressedVector<double>>(
              "[0., 1., 0., 2.]")
              .nonZeros() == 2);
  }
  {
    INFO("StaticMatrix");
    test_serialization(blaze::StaticMatrix<double, 2, 3, blaze::columnMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    test_serialization(blaze::StaticMatrix<double, 2, 3, blaze::rowMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::StaticMatrix<double, 2, 3, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::StaticMatrix<double, 2, 3, blaze::columnMajor>{{0., 1., 2.},
                                                                {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::StaticMatrix<double, 2, 3, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::StaticMatrix<double, 2, 3, blaze::rowMajor>{{0., 1., 2.},
                                                             {3., 0., 4.}});
  }
  {
    INFO("DynamicMatrix");
    test_serialization(blaze::DynamicMatrix<double, blaze::columnMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    test_serialization(blaze::DynamicMatrix<double, blaze::rowMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<double, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::DynamicMatrix<double, blaze::columnMajor>{{0., 1., 2.},
                                                           {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<double, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::DynamicMatrix<double, blaze::rowMajor>{{0., 1., 2.},
                                                        {3., 0., 4.}});
    // Test `write_csv`
    std::stringstream ss{};
    blaze::DynamicMatrix<double, blaze::columnMajor> matrix{{0., 1., 2.},
                                                            {3., 0., 4.}};
    write_csv(ss, matrix);
    CHECK(ss.str() == "0,1,2\n3,0,4\n");
  }
  {
    INFO("Complex DynamicMatrix");
    using namespace std::complex_literals;
    test_serialization(
        blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>{
            {0. + 1.i, 1. + 3.i, 2. + 2.i}, {3. + 4.i, 0. - 1.i, 4. + 5.i}});
    test_serialization(
        blaze::DynamicMatrix<std::complex<double>, blaze::rowMajor>{
            {0. + 1.i, 1. + 3.i, 2. + 2.i}, {3. + 4.i, 0. - 1.i, 4. + 5.i}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>>(
              "[[[0., 1.], [1., 3.], [2., 2.]], [[3., 4.], [0., -1.], [4., "
              "5.]]]") ==
          blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>{
              {0. + 1.i, 1. + 3.i, 2. + 2.i}, {3. + 4.i, 0. - 1.i, 4. + 5.i}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<std::complex<double>, blaze::rowMajor>>(
              "[[[0., 1.], [1., 3.], [2., 2]], [[3., 4.], [0., -1.], [4., "
              "5.]]]") ==
          blaze::DynamicMatrix<std::complex<double>, blaze::rowMajor>{
              {0. + 1.i, 1. + 3.i, 2. + 2.i}, {3. + 4.i, 0. - 1.i, 4. + 5.i}});
    // Test `write_csv`
    std::stringstream ss{};
    blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor> matrix{
        {{0. + 1.i, 1. + 3.i, 2. + 2.i}, {3. + 4.i, 0. - 1.i, 4. + 5.i}}};
    write_csv(ss, matrix);
    CHECK(ss.str() == "(0,1),(1,3),(2,2)\n(3,4),(0,-1),(4,5)\n");
  }
  {
    INFO("CompressedMatrix");
    test_serialization(blaze::CompressedMatrix<double, blaze::columnMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(serialize_and_deserialize(
              blaze::CompressedMatrix<double, blaze::columnMajor>{{0., 1., 2.},
                                                                  {3., 0., 4.}})
              .nonZeros() == 4);
    test_serialization(blaze::CompressedMatrix<double, blaze::rowMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(serialize_and_deserialize(
              blaze::CompressedMatrix<double, blaze::rowMajor>{{0., 1., 2.},
                                                               {3., 0., 4.}})
              .nonZeros() == 4);
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::CompressedMatrix<double, blaze::columnMajor>{{0., 1., 2.},
                                                              {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]")
              .nonZeros() == 4);
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::CompressedMatrix<double, blaze::rowMajor>{{0., 1., 2.},
                                                           {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]")
              .nonZeros() == 4);
  }
}
