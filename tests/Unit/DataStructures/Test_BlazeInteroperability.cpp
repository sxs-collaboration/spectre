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
#include "Utilities/MakeWithValue.hpp"

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
