// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/Deferred.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
/// [functions_used]
struct func {
  double operator()() const { return 8.2; }
};

double dummy() { return 6.7; }

struct func2 {
  double operator()(const double t) const { return t; }
};

double lazy_function(const double t) { return 10.0 * t; }

void mutate_function(const gsl::not_null<double*> t, const double t0) {
  *t = t0;
}

void mutate_function_vector(const gsl::not_null<std::vector<double>*> t,
                            const std::vector<double>& t0) {
  if (t->size() != t0.size()) {
    t->resize(t0.size(), 0.0);
  }
  // Check the size again just to be sure the resize above happened.
  CHECK(t->size() == t0.size());
  for (size_t i = 0; i < t->size(); ++i) {
    t->operator[](i) = 10.0 * t0[i];
  }
}
/// [functions_used]

void simple_deferred() {
  /// [deferred_with_update]
  auto obj = Deferred<double>(3.8);
  CHECK(obj.evaluated());
  CHECK(3.8 == obj.get());
  auto& obj_val = obj.mutate();
  CHECK(3.8 == obj_val);
  obj_val = 5.0;
  CHECK(5.0 == obj.get());
  /// [deferred_with_update]
  CHECK(obj.evaluated());
}

void single_call_deferred() {
  /// [make_deferred_with_function_object]
  auto def = make_deferred<double>(func{});
  CHECK_FALSE(def.evaluated());
  CHECK(8.2 == def.get());
  /// [make_deferred_with_function_object]
  CHECK(def.evaluated());

  /// [make_deferred_with_function]
  auto def2 = make_deferred<double>(dummy);
  CHECK(6.7 == def2.get());
  /// [make_deferred_with_function]

  const auto function_name = dummy;
  auto def3 = make_deferred<double>(function_name);
  CHECK(6.7 == def3.get());
}

void deferred_as_argument_to_deferred() {
  /// [make_deferred_with_deferred_arg]
  auto def2 = make_deferred<double>(func2{}, 6.82);
  auto def3 = make_deferred<double>(lazy_function, def2);
  CHECK(68.2 == def3.get());
  CHECK(6.82 == def2.get());
  /// [make_deferred_with_deferred_arg]
}

void mutating_deferred() {
  auto def2 = make_deferred<double>(func2{}, 6.82);
  auto def_mutate = make_deferred<double>(mutate_function, def2);
  CHECK(6.82 == def_mutate.get());

  auto def_mutate_vector = make_deferred<std::vector<double>>(
      mutate_function_vector, std::vector<double>{2.3, 8.9, 7.8});
  CHECK((std::vector<double>{23.0, 89.0, 78.0}) == def_mutate_vector.get());
}

struct counting_func {
  double operator()(double a = 1.0) const {
    count++;
    return 8.2 * a;
  }
  static int count;
};

int counting_func::count = 0;

template <typename T, typename Lambda>
Deferred<T> serialize_and_deserialize_deferred(Deferred<T>& deferred,
                                               Lambda make_def) {
  PUP::sizer sizer;
  deferred.pack_unpack_lazy_function(sizer);
  std::vector<char> data(sizer.size());
  PUP::toMem writer(data.data());
  deferred.pack_unpack_lazy_function(writer);

  PUP::fromMem reader(data.data());
  Deferred<T> return_deferred = make_def();
  return_deferred.pack_unpack_lazy_function(reader);
  return return_deferred;
}

void serialization() {
  INFO("Testing unevaluated Deferred serialization...");
  {
    auto counting_func_deferred = make_deferred<double>(counting_func{});
    CHECK(counting_func::count == 0);
    auto counting_func_deferred_sent = serialize_and_deserialize_deferred(
        counting_func_deferred,
        []() { return make_deferred<double>(counting_func{}); });
    CHECK(counting_func::count == 0);
    CHECK(counting_func_deferred.get() == 8.2);
    CHECK(counting_func::count == 1);
    CHECK(counting_func_deferred_sent.get() == 8.2);
    CHECK(counting_func::count == 2);
    CHECK(&counting_func_deferred_sent.get() != &counting_func_deferred.get());
    counting_func::count = 0;
  }
  {
    auto counting_func_deferred = make_deferred<double>(counting_func{}, 2.0);
    CHECK(counting_func::count == 0);
    auto counting_func_deferred_sent = serialize_and_deserialize_deferred(
        counting_func_deferred,
        []() { return make_deferred<double>(counting_func{}, 2.0); });
    CHECK(counting_func::count == 0);
    CHECK(counting_func_deferred.get() == 16.4);
    CHECK(counting_func::count == 1);
    CHECK(counting_func_deferred_sent.get() == 16.4);
    CHECK(counting_func::count == 2);
    CHECK(&counting_func_deferred_sent.get() != &counting_func_deferred.get());
    counting_func::count = 0;
  }
  INFO("Testing evaluated Deferred serialization...");
  {
    CHECK(counting_func::count == 0);
    auto counting_func_deferred = make_deferred<double>(counting_func{});
    CHECK(counting_func::count == 0);
    CHECK(counting_func_deferred.get() == 8.2);
    CHECK(counting_func::count == 1);
    auto counting_func_deferred_sent = serialize_and_deserialize_deferred(
        counting_func_deferred,
        []() { return make_deferred<double>(counting_func{}); });
    CHECK(counting_func_deferred.get() == 8.2);
    CHECK(counting_func::count == 1);
    CHECK(counting_func_deferred_sent.get() == 8.2);
    CHECK(counting_func::count == 1);
    CHECK(&counting_func_deferred_sent.get() != &counting_func_deferred.get());
    counting_func::count = 0;
  }
  {
    CHECK(counting_func::count == 0);
    auto counting_func_deferred = make_deferred<double>(counting_func{}, 2.0);
    CHECK(counting_func::count == 0);
    CHECK(counting_func_deferred.get() == 16.4);
    CHECK(counting_func::count == 1);
    auto counting_func_deferred_sent = serialize_and_deserialize_deferred(
        counting_func_deferred,
        []() { return make_deferred<double>(counting_func{}, 2.0); });
    CHECK(counting_func_deferred.get() == 16.4);
    CHECK(counting_func::count == 1);
    CHECK(counting_func_deferred_sent.get() == 16.4);
    CHECK(counting_func::count == 1);
    CHECK(&counting_func_deferred_sent.get() != &counting_func_deferred.get());
  }
}

// Test that the special case of a Deferred<const T&>, which is used for data
// that is pointing to elsewhere (such as a global cache).
void test_deferred_const_ref() {
  const double a_value = 5.0;
  const Deferred<const double*> a(&a_value);
  const auto helper = [](const double* const t) -> const double& { return *t; };
  auto deferred_double = make_deferred<const double&>(helper, a);
  CHECK(deferred_double.get() == 5.0);
  CHECK(&deferred_double.get() == &a_value);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Deferred",
                  "[DataStructures][Unit]") {
  simple_deferred();
  single_call_deferred();
  deferred_as_argument_to_deferred();
  mutating_deferred();
  serialization();
  test_deferred_const_ref();
}

// [[OutputRegex, Cannot mutate a computed Deferred]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Deferred.FailAlter",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto def = make_deferred<double>(func{});
  def.mutate();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Cannot send a Deferred that's not a lazily evaluated
// function]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Deferred.PupNonfunction",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  Deferred<double> deferred{3.89};
  auto data = std::make_unique<char[]>(10);
  PUP::fromMem p{static_cast<const void*>(data.get())};
  deferred.pack_unpack_lazy_function(p);
}
