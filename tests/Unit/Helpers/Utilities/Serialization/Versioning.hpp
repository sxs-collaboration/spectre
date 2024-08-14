// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace TestHelpers {
/// \brief Collection of classes and functions for testing serialization
namespace serialization {
namespace versioning_detail {
std::vector<std::pair<std::string, std::vector<std::byte>>> read_serializations(
    const std::string& filename);

void write_serialization(const std::string& filename, const std::string& label,
                         const std::vector<std::byte>& serialization);

template <typename Compare>
struct ToCompare {
  const Compare& operator()(const Compare& x) const { return x; }

  template <typename Base>
  const Compare& operator()(const std::unique_ptr<Base>& x) const {
    return dynamic_cast<const Compare&>(*x);
  }
};

template <>
struct ToCompare<void> {
  template <typename Serialize>
  const Serialize& operator()(const Serialize& x) const {
    static_assert(not tt::is_a_v<std::unique_ptr, Serialize>,
                  "Cannot compare unique_ptrs.  Pass a template argument to "
                  "test_versioning to set type to compare.");
    return x;
  }
};
}  // namespace versioning_detail

/// Test serialization of a versioned class against old versions.
///
/// Old serializations of the class are stored in \p filename
/// (relative to the unit test source directory).  Each stored version
/// has a label, where the current label must match \p current_label.
/// This function checks that
///
/// * The last entry matches \p current_label and matches the result
///   of serializing \p current_object.
///
/// * Old lines in the file are successfully deserialized and give
///   expected values.
///
/// By default, old versions are expected to deserialize to \p
/// current_object, but if the object represented by the serialized
/// entries has changed (for example, because the \p current_object
/// has newer options enabled), objects can be passed in \p
/// old_objects, keyed by the last version they should match.  If
/// deserialization support is dropped up to a specific entry, an
/// error can be checked for by instead adding a string to \p
/// old_objects.
///
/// By default, objects are directly compared for equality.  If the
/// objects being tested are stored in `std::unique_ptr`s the \p
/// Compare template parameter can be passed in to dereference and
/// `dynamic_cast` the contained value before comparing.
///
/// If \p generate_new_entry is true and \p current_label and \p
/// current_object do not match the last serialization in the file, a
/// new entry will be appended.
template <typename Compare = void, typename Serialize>
void test_versioning(
    const std::string& filename, const std::string& current_label,
    const Serialize& current_object,
    const std::unordered_map<std::string, std::variant<Serialize, std::string>>&
        old_objects = {},
    const bool generate_new_entry = false) {
  const versioning_detail::ToCompare<Compare> to_compare{};

  {
    INFO("Current version should not be listed in old_objects.");
    CHECK(old_objects.count(current_label) == 0);
  }

  const auto serializations_to_test =
      versioning_detail::read_serializations(filename);

  std::unordered_set<std::string> unused_labels{};
  for (const auto& old_entry : old_objects) {
    unused_labels.insert(old_entry.first);
  }

  std::variant<const Serialize*, const std::string*> expected = &current_object;
  for (auto serialization_test = serializations_to_test.rbegin();
       serialization_test != serializations_to_test.rend();
       ++serialization_test) {
    const auto& [version, serialization] = *serialization_test;
    CAPTURE(version);
    if (const auto old_object = old_objects.find(version);
        old_object != old_objects.end()) {
      unused_labels.erase(version);
      std::visit([&](const auto& old) { expected = &old; }, old_object->second);
    }

    if (std::holds_alternative<const Serialize*>(expected)) {
      CHECK(to_compare(deserialize<Serialize>(serialization.data())) ==
            to_compare(*std::get<const Serialize*>(expected)));
    } else {
      CHECK_THROWS_WITH(deserialize<Serialize>(serialization.data()),
                        Catch::Matchers::ContainsSubstring(
                            *std::get<const std::string*>(expected)));
    }
  }
  {
    CAPTURE(unused_labels);
    CHECK(unused_labels.empty());
  }

  const std::vector<char> current_serialization_chars =
      serialize(current_object);
  std::vector<std::byte> current_serialization(
      current_serialization_chars.size());
  std::transform(current_serialization_chars.begin(),
                 current_serialization_chars.end(),
                 current_serialization.begin(),
                 [](const char c) { return static_cast<std::byte>(c); });

  if (generate_new_entry) {
    {
      INFO("Entry already present.");
      REQUIRE(
          (serializations_to_test.empty() or
           (current_label != serializations_to_test.back().first and
            current_serialization != serializations_to_test.back().second)));
    }
    REQUIRE(to_compare(deserialize<Serialize>(current_serialization.data())) ==
            to_compare(current_object));
    versioning_detail::write_serialization(filename, current_label,
                                           current_serialization);
  } else {
    REQUIRE(not serializations_to_test.empty());
    CHECK(current_label == serializations_to_test.back().first);
    CHECK(current_serialization == serializations_to_test.back().second);
  }
}
}  // namespace serialization
}  // namespace TestHelpers
