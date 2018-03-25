// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <stddef.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace Test_Classes {
struct DerivedInPupStlCpp11;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
struct Base : public PUP::able {
  using creatable_classes = tmpl::list<Test_Classes::DerivedInPupStlCpp11>;
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_abstract(Base);  // NOLINT
};
#pragma GCC diagnostic pop

struct DerivedInPupStlCpp11 : public Base {
  explicit DerivedInPupStlCpp11(std::vector<double> vec)
      : vec_(std::move(vec)) {}
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_decl_base_template(Base,  // NOLINT
                                     DerivedInPupStlCpp11);
  explicit DerivedInPupStlCpp11(CkMigrateMessage* /* m */) {}
  void pup(PUP::er& p) override {
    Base::pup(p);
    p | vec_;
  }

  const auto& get() const { return vec_; }

  friend bool operator==(const DerivedInPupStlCpp11& lhs,
                         const DerivedInPupStlCpp11& rhs) {
    return lhs.vec_ == rhs.vec_;
  }

 private:
  std::vector<double> vec_;
};

}  // namespace Test_Classes

namespace {
enum class eDummyEnum { test1, test2 };
}  // namespace

SPECTRE_TEST_CASE("Unit.Serialization.unordered_map", "[Serialization][Unit]") {
  std::unordered_map<std::string, double> um;
  um["aaa"] = 1.589;
  um["bbb"] = -10.7392;
  test_serialization(um);
}

SPECTRE_TEST_CASE("Unit.Serialization.enum", "[Serialization][Unit]") {
  eDummyEnum e = eDummyEnum::test2;
  test_serialization(e);
}

/// [example_serialize_comparable]
SPECTRE_TEST_CASE("Unit.Serialization.tuple", "[Serialization][Unit]") {
  std::unordered_map<std::string, double> um;
  um["aaa"] = 1.589;
  um["bbb"] = -10.7392;
  auto test_tuple = std::make_tuple<int, double, std::string,
                                    std::unordered_map<std::string, double>>(
      2, 0.57, "blah", std::move(um));
  test_serialization(test_tuple);
}
/// [example_serialize_comparable]

SPECTRE_TEST_CASE("Unit.Serialization.array", "[Serialization][Unit]") {
  auto t = make_array(1.0, 3.64, 9.23);
  test_serialization(t);

  auto t2 =
      make_array(std::vector<double>{1, 4, 8}, std::vector<double>{7, 9, 4});
  test_serialization(t2);
}

SPECTRE_TEST_CASE("Unit.Serialization.deque", "[Serialization][Unit]") {
  std::deque<double> t{1.0, 3.64, 9.23};
  test_serialization(t);
}

SPECTRE_TEST_CASE("Unit.Serialization.unordered_set", "[Serialization][Unit]") {
  std::unordered_set<size_t> test_set = {1, 2, 5, 100};
  test_serialization(test_set);
}

SPECTRE_TEST_CASE("Unit.Serialization.unordered_set.empty",
                  "[Serialization][Unit]") {
  std::unordered_set<size_t> test_set{};
  test_serialization(test_set);
}

SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.double",
                  "[Serialization][Unit]") {
  auto test_unique_ptr = std::make_unique<double>(3.8273);
  // clang-tidy: false positive use after free
  CHECK(3.8273 == *serialize_and_deserialize(test_unique_ptr));  // NOLINT
}

/// [example_serialize_derived]
SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.abstract_base",
                  "[Serialization][Unit]") {
  test_serialization_via_base<Test_Classes::Base,
                              Test_Classes::DerivedInPupStlCpp11>(
      std::vector<double>{-1, 12.3, -7, 8});
}
/// [example_serialize_derived]

SPECTRE_TEST_CASE("Unit.Serialization.unique_ptr.nullptr",
                  "[Serialization][Unit]") {
  std::unique_ptr<double> derived_ptr = nullptr;
  auto blah = serialize_and_deserialize(derived_ptr);
  CHECK(nullptr == blah);
}

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Test_Classes::DerivedInPupStlCpp11::my_PUP_ID = 0;  // NOLINT
/// \endcond
