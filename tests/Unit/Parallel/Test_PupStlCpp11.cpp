// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Parallel/ParallelTestClasses.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
enum class eDummyEnum { test1, test2 };
}  // namespace

SPECTRE_TEST_CASE("Unit.Serialization.unordered_map", "[Serialization][Unit]") {
  std::unordered_map<std::string, double> um;
  um["aaa"] = 1.589;
  um["bbb"] = -10.7392;
  CHECK(um == serialize_and_deserialize(um));
}

SPECTRE_TEST_CASE("Unit.Serialization.enum", "[Serialization][Unit]") {
  eDummyEnum test1 = eDummyEnum::test2;
  CHECK(test1 == serialize_and_deserialize(test1));
}

/// [example_serialize_copyable]
SPECTRE_TEST_CASE("Unit.Serialization.tuple", "[Serialization][Unit]") {
  std::unordered_map<std::string, double> um;
  um["aaa"] = 1.589;
  um["bbb"] = -10.7392;
  auto test_tuple = std::make_tuple<int, double, std::string,
                                    std::unordered_map<std::string, double>>(
      2, 0.57, "blah", std::move(um));
  CHECK(test_tuple == serialize_and_deserialize(test_tuple));
}
/// [example_serialize_copyable]

SPECTRE_TEST_CASE("Unit.Serialization.array", "[Serialization][Unit]") {
  auto t = make_array(1.0, 3.64, 9.23);
  CHECK(t == serialize_and_deserialize(t));

  auto t2 =
      make_array(std::vector<double>{1, 4, 8}, std::vector<double>{7, 9, 4});
  CHECK(t2 == serialize_and_deserialize(t2));
}

SPECTRE_TEST_CASE("Unit.Serialization.unordered_set", "[Serialization][Unit]") {
  std::unordered_set<size_t> test_set = {1, 2, 5, 100};
  CHECK(test_set == serialize_and_deserialize(test_set));
}

SPECTRE_TEST_CASE("Unit.Serialization.unordered_set.empty",
                  "[Serialization][Unit]") {
  std::unordered_set<size_t> test_set{};
  CHECK(test_set == serialize_and_deserialize(test_set));
}

SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.double",
                  "[Serialization][Unit]") {
  auto test_unique_ptr = std::make_unique<double>(3.8273);
  // clang-tidy: false positive use after free
  CHECK(3.8273 == *serialize_and_deserialize(test_unique_ptr));  // NOLINT
}

SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.abstract_base",
                  "[Serialization][Unit]") {
  Test_Classes::DerivedInPupStlCpp11 derived({-1, 12.3, -7, 8});
  std::unique_ptr<Test_Classes::Base> derived_ptr =
      std::make_unique<Test_Classes::DerivedInPupStlCpp11>(
          std::vector<double>{-1, 12.3, -7, 8});
  std::unique_ptr<Test_Classes::Base> serialized_ptr =
      serialize_and_deserialize(derived_ptr);
  CHECK(nullptr != dynamic_cast<Test_Classes::DerivedInPupStlCpp11*>(
                       serialized_ptr.get()));
  CHECK(derived == dynamic_cast<const Test_Classes::DerivedInPupStlCpp11&>(
                       *serialized_ptr));
}

SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.nullptr",
                  "[Serialization][Unit]") {
  std::unique_ptr<double> derived_ptr = nullptr;
  auto blah = serialize_and_deserialize(derived_ptr);
  CHECK(nullptr == blah);
}
