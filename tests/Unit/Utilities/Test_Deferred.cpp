// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/Deferred.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
/// [functions_used]
struct func {
  double operator()() const { return 8.2; }
};

double dummy() { return 6.7; }

struct func2 {
  double operator()(const double& t) const { return t; }
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

void mutate_function_vector_evil(const gsl::not_null<std::vector<double>*> t,
                                 const std::vector<double>& t0) {
  if (t->size() != t0.size()) {
    t->resize(t0.size(), 0.0);
  }
  // Check the size again just to be sure the resize above happened.
  CHECK(t->size() == t0.size());
  for (size_t i = 0; i < t->size(); ++i) {
    t->operator[](i) += 10.0 * t0[i];
  }
}

void simple_deferred() {
  /// [deferred_with_update]
  auto obj = Deferred<double>(3.8);
  CHECK(3.8 == obj.get());
  auto& obj_val = obj.mutate();
  CHECK(3.8 == obj_val);
  obj_val = 5.0;
  CHECK(5.0 == obj.get());
  /// [deferred_with_update]
}

void single_call_deferred() {
  /// [make_deferred_with_function_object]
  auto def = make_deferred<double>(func{});
  CHECK(8.2 == def.get());
  /// [make_deferred_with_function_object]

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

void update_deferred() {
  auto lazy_deferred = make_deferred<double>(lazy_function, 3.4);
  CHECK(lazy_deferred.get() == 34.);
  update_deferred_args(make_not_null(&lazy_deferred), lazy_function, 5.5);
  CHECK(lazy_deferred.get() == 55.);

  /// [update_args_of_deferred_deduced_fp]
  auto mutate_deferred = make_deferred<std::vector<double>>(
      mutate_function_vector, std::vector<double>{1.3, 7.8, 9.8});
  CHECK(mutate_deferred.get() == (std::vector<double>{13., 78., 98.}));
  update_deferred_args(make_not_null(&mutate_deferred), mutate_function_vector,
                       std::vector<double>{10., 70., 90.});
  CHECK(mutate_deferred.get() == (std::vector<double>{100., 700., 900.}));
  /// [update_args_of_deferred_deduced_fp]

  /// [update_args_of_deferred_specified_fp]
  update_deferred_args<std::vector<double>, decltype(mutate_function_vector)>(
      &mutate_deferred, std::vector<double>{20., 8., 9.});
  CHECK(mutate_deferred.get() == (std::vector<double>{200., 80., 90.}));
  /// [update_args_of_deferred_specified_fp]

  auto mutate_deferred_evil = make_deferred<std::vector<double>>(
      mutate_function_vector_evil, std::vector<double>{1.3, 7.8, 9.8});
  CHECK(mutate_deferred_evil.get() == (std::vector<double>{13., 78., 98.}));
  const double* const initial_pointer = mutate_deferred_evil.get().data();
  update_deferred_args(make_not_null(&mutate_deferred_evil),
                       mutate_function_vector_evil,
                       std::vector<double>{10., 70., 90.});
  CHECK(mutate_deferred_evil.get() == (std::vector<double>{113., 778., 998.}));
  CHECK(initial_pointer == mutate_deferred_evil.get().data());

  update_deferred_args<std::vector<double>,
                       decltype(mutate_function_vector_evil)>(
      &mutate_deferred_evil, std::vector<double>{10., 70., 90.});
  CHECK(mutate_deferred_evil.get() ==
        (std::vector<double>{213., 1478., 1898.}));
  CHECK(initial_pointer == mutate_deferred_evil.get().data());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Deferred", "[Utilities][Unit]") {
  simple_deferred();
  single_call_deferred();
  deferred_as_argument_to_deferred();
  mutating_deferred();
  update_deferred();
}

// [[OutputRegex, Cannot cast the Deferred class to:
// Deferred_detail::deferred_assoc_state]]
SPECTRE_TEST_CASE("Unit.Utilities.Deferred.UpdateArgsError0",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  auto lazy_deferred = make_deferred<double>(lazy_function, 3.4);
  update_deferred_args<double>(&lazy_deferred, lazy_function, 5);
}

// [[OutputRegex, Cannot cast the Deferred class to:
// Deferred_detail::deferred_assoc_state]]
SPECTRE_TEST_CASE("Unit.Utilities.Deferred.UpdateArgsError1",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  auto lazy_deferred = make_deferred<double>(lazy_function, 3.4);
  update_deferred_args<double, decltype(lazy_function)>(&lazy_deferred, 5);
}

// [[OutputRegex, Cannot cast the Deferred class to:
// Deferred_detail::deferred_assoc_state]]
SPECTRE_TEST_CASE("Unit.Utilities.Deferred.UpdateArgsError2",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  auto lazy_deferred = make_deferred<double>(lazy_function, 3.4);
  update_deferred_args<double, decltype(mutate_function)>(&lazy_deferred, 5.5);
}

// [[OutputRegex, Cannot mutate a computed Deferred]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Deferred.FailAlter",
                               "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto def = make_deferred<double>(func{});
  auto& mutate = def.mutate();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
