// Distributed under the MIT License.
// See LICENSE.txt for detai

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct AnalyticSolutionTest {
  template <typename T>
  struct Var1 {
    using type = Scalar<T>;
  };
  template <typename T>
  struct Var2 {
    using type = tnsr::I<T, 3>;
  };

  AnalyticSolutionTest(const double a, const std::array<double, 3>& b)
      : a_(a), b_(b) {}

  template <typename T>
  tuples::TaggedTuple<Var1<T>, Var2<T>> solution(const tnsr::i<T, 3>& x,
                                                 const double t) const {
    auto sol =
        make_with_value<tuples::TaggedTuple<Var1<T>, Var2<T>>>(x.get(0), 0.);
    auto& scalar = tuples::get<Var1<T>>(sol);
    auto& vector = tuples::get<Var2<T>>(sol);

    for (size_t i = 0; i < 3; ++i) {
      scalar.get() += x.get(i) * gsl::at(b_, i);
      vector.get(i) = a_ * x.get(i) - gsl::at(b_, i) * t;
    }
    scalar.get() += a_ - t;
    return sol;
  }

 private:
  double a_;
  std::array<double, 3> b_;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Pypp.AnalyticSolution", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Framework/Tests/"};
  const double a = 4.3;
  const std::array<double, 3> b{{-1.3, 5.6, -0.2}};
  AnalyticSolutionTest solution{a, b};
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      &AnalyticSolutionTest::solution<double>, solution, "PyppPyTests",
      {"check_solution_scalar", "check_solution_vector"}, {{{-10.0, 10.0}}},
      std::make_tuple(a, b), a);
  pypp::check_with_random_values<1>(
      &AnalyticSolutionTest::solution<DataVector>, solution, "PyppPyTests",
      {"check_solution_scalar", "check_solution_vector"}, {{{-10.0, 10.0}}},
      std::make_tuple(a, b), used_for_size);
}
