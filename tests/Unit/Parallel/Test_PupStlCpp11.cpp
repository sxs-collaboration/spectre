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
#include <vector>
#include <utility>

#include "Framework/TestHelpers.hpp"
#include "Parallel/CharmPupable.hpp"
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

SPECTRE_TEST_CASE("Unit.Serialization.PupStlCpp11", "[Serialization][Unit]") {
  /// [example_serialize_comparable]
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
  /// [example_serialize_comparable]
  /// [example_serialize_derived]
  {
    INFO("unique_ptr.abstract_base");
    test_serialization_via_base<Test_Classes::Base,
                                Test_Classes::DerivedInPupStlCpp11>(
                                    std::vector<double>{-1, 12.3, -7, 8});
  }
  /// [example_serialize_derived]
}

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Test_Classes::DerivedInPupStlCpp11::my_PUP_ID = 0;  // NOLINT
/// \endcond
