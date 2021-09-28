// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <deque>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/TMPL.hpp"

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

struct PairComparator {
  bool operator()(std::pair<int, double> lhs,
                  std::pair<int, double> rhs) const noexcept {
    return lhs.first < rhs.first or
           (lhs.first == rhs.first and lhs.second < rhs.second);
  }
};

SPECTRE_TEST_CASE("Unit.Serialization.PupStlCpp11", "[Serialization][Unit]") {
  // [example_serialize_comparable]
  {
    INFO("tuple");
    std::unordered_map<std::string, double> um;
    um["aaa"] = 1.589;
    um["bbb"] = -10.7392;
    auto test_tuple = std::make_tuple<int, double, std::string,
                                      std::unordered_map<std::string, double>>(
        2, 0.57, "blah", std::move(um));
    test_serialization(test_tuple);
  }
  // [example_serialize_comparable]
  // [example_serialize_derived]
  {
    INFO("unique_ptr.abstract_base");
    test_serialization_via_base<Test_Classes::Base,
                                Test_Classes::DerivedInPupStlCpp11>(
        std::vector<double>{-1, 12.3, -7, 8});
  }
  // [example_serialize_derived]
  {
    // note the `serialize_and_deserialize` utilities don't work here because
    // atomic specifically has no copy constructor -- it must be serialized
    // directly into the target object.
    INFO("atomic int");
    const std::atomic<int> to_serialize{3};
    std::atomic<int> serialization_target{234};
    const auto serialized_data = serialize<std::atomic<int>>(to_serialize);
    CHECK(to_serialize != serialization_target);
    PUP::fromMem reader(serialized_data.data());
    reader | serialization_target;
    CHECK(serialization_target == to_serialize);
  }
  {
    INFO("atomic double");
    const std::atomic<double> to_serialize{3.4};
    std::atomic<double> serialization_target{23.5};
    const auto serialized_data = serialize<std::atomic<double>>(to_serialize);
    CHECK(to_serialize != serialization_target);
    PUP::fromMem reader(serialized_data.data());
    reader | serialization_target;
    CHECK(serialization_target == to_serialize);
  }
  {
    INFO("map with custom comparator");
    std::map<std::pair<int, double>, double, PairComparator> map_to_serialize;
    map_to_serialize.insert({std::make_pair(1, 2.0), 3.0});
    map_to_serialize.insert({std::make_pair(3, 1.0), 1.5});
    map_to_serialize.insert({std::make_pair(2, 6.0), 10.2});
    std::map<std::pair<int, double>, double, PairComparator>
        serialization_target;
    CHECK(map_to_serialize != serialization_target);
    PUP::sizer sizer;
    pup_override(sizer, map_to_serialize);
    std::vector<char> data(sizer.size());
    PUP::toMem writer(data.data());
    pup_override(writer, map_to_serialize);
    PUP::fromMem reader(data.data());
    pup_override(reader, serialization_target);
    CHECK(serialization_target == map_to_serialize);
  }
}

// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Test_Classes::DerivedInPupStlCpp11::my_PUP_ID = 0;  // NOLINT
