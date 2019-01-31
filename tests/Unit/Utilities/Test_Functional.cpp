// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>  // IWYU pragma: keep
#include <complex>
#include <cstddef>
#include <random>
#include <tuple>
#include <utility>

#include "ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_include <algorithm>

namespace funcl {
using std::imag;
using std::real;
using std::max;
using std::min;

// clang-tidy : suppress warning from no-paren on macro parameter, macro doesn't
// work when it's in parens. Same for all subsequent macros.
#define MAKE_UNARY_TEST(STRUCTNAME, FUNC)                                  \
  struct TestFuncEval##STRUCTNAME {                                        \
    template <typename T, typename DistT, typename UniformGen>             \
    void operator()(UniformGen gen, UniformCustomDistribution<DistT> dist, \
                    T /*meta*/) noexcept {                                 \
      auto val = make_with_random_values<typename T::type>(                \
          make_not_null(&gen), make_not_null(&dist));                      \
      CHECK(STRUCTNAME<>{}(val) == FUNC(val)); /*NOLINT*/                  \
    }                                                                      \
  }

#define MAKE_BINARY_TEST(STRUCTNAME, FUNC)                                   \
  struct TestFuncEval##STRUCTNAME {                                          \
    template <typename T1, typename T2, typename DistT1, typename DistT2,    \
              typename UniformGen>                                           \
    void operator()(UniformGen gen, UniformCustomDistribution<DistT1> dist1, \
                    UniformCustomDistribution<DistT2> dist2, T1 /*meta*/,    \
                    T2 /*meta*/) noexcept {                                  \
      auto val1 = make_with_random_values<typename T1::type>(                \
          make_not_null(&gen), make_not_null(&dist1));                       \
      auto val2 = make_with_random_values<typename T2::type>(                \
          make_not_null(&gen), make_not_null(&dist2));                       \
      CHECK(STRUCTNAME<>{}(val1, val2) == FUNC(val1, val2)); /*NOLINT*/      \
    }                                                                        \
  }

#define MAKE_BINARY_OP_TEST(STRUCTNAME, OP)                                  \
  struct TestFuncEval##STRUCTNAME {                                          \
    template <typename T1, typename T2, typename DistT1, typename DistT2,    \
              typename UniformGen>                                           \
    void operator()(UniformGen gen, UniformCustomDistribution<DistT1> dist1, \
                    UniformCustomDistribution<DistT2> dist2, T1 /*meta*/,    \
                    T2 /*meta*/) noexcept {                                  \
      auto val1 = make_with_random_values<typename T1::type>(                \
          make_not_null(&gen), make_not_null(&dist1));                       \
      auto val2 = make_with_random_values<typename T2::type>(                \
          make_not_null(&gen), make_not_null(&dist2));                       \
      CHECK(STRUCTNAME<>{}(val1, val2) == val1 OP val2); /*NOLINT*/          \
    }                                                                        \
  }

#define MAKE_BINARY_INPLACE_TEST(STRUCTNAME, OP, TESTOP)                       \
  struct TestFuncEval##STRUCTNAME {                                            \
    template <typename T1, typename T2, typename DistT1, typename DistT2,      \
              typename UniformGen,                                             \
              Requires<cpp17::is_same_v<                                       \
                  typename T1::type,                                           \
                  decltype(std::declval<typename T1::type>() TESTOP            \
                               std::declval<typename T2::type>())>> = nullptr> \
    void operator()(UniformGen gen, UniformCustomDistribution<DistT1> dist1,   \
                    UniformCustomDistribution<DistT2> dist2, T1 /*meta*/,      \
                    T2 /*meta*/) noexcept {                                    \
      auto val1 = make_with_random_values<typename T1::type>(                  \
          make_not_null(&gen), make_not_null(&dist1));                         \
      auto val2 = make_with_random_values<typename T2::type>(                  \
          make_not_null(&gen), make_not_null(&dist2));                         \
      auto val1_copy = val1;                                                   \
      auto result = val1_copy TESTOP val2;                                     \
      STRUCTNAME<>{}(val1, val2); /*NOLINT*/                                   \
      CHECK(val1 == result);                                                   \
    }                                                                          \
    /* SFINAE to avoid testing illegal combinations like */                    \
    /* double += complex<double> in addition to vise-versa. */                 \
    /* Below no-op happens when inplace cannot be called. */                   \
    template <typename T1, typename T2, typename DistT1, typename DistT2,      \
              typename UniformGen,                                             \
              Requires<not cpp17::is_same_v<                                   \
                  typename T1::type,                                           \
                  decltype(std::declval<typename T1::type>() TESTOP            \
                               std::declval<typename T2::type>())>> = nullptr> \
    void operator()(UniformGen /*gen*/,                                        \
                    UniformCustomDistribution<DistT1> /*dist1*/,               \
                    UniformCustomDistribution<DistT2> /*dist2*/, T1 /*meta*/,  \
                    T2 /*meta*/) noexcept {}                                   \
  }

namespace {

MAKE_UNARY_TEST(Abs, abs);
MAKE_UNARY_TEST(Acos, acos);
MAKE_UNARY_TEST(Acosh, acosh);
MAKE_UNARY_TEST(Asin, asin);
MAKE_UNARY_TEST(Asinh, asinh);
MAKE_UNARY_TEST(Atan, atan);
MAKE_UNARY_TEST(Atanh, atanh);
MAKE_UNARY_TEST(Cbrt, cbrt);
MAKE_UNARY_TEST(Cos, cos);
MAKE_UNARY_TEST(Cosh, cosh);
MAKE_UNARY_TEST(Erf, erf);
MAKE_UNARY_TEST(Exp, exp);
MAKE_UNARY_TEST(Exp2, exp2);
MAKE_UNARY_TEST(Fabs, fabs);
MAKE_UNARY_TEST(Imag, imag);
MAKE_UNARY_TEST(InvCbrt, invcbrt);
MAKE_UNARY_TEST(InvSqrt, invsqrt);
MAKE_UNARY_TEST(Log, log);
MAKE_UNARY_TEST(Log10, log10);
MAKE_UNARY_TEST(Log2, log2);
MAKE_UNARY_TEST(Real, real);
MAKE_UNARY_TEST(Sin, sin);
MAKE_UNARY_TEST(Sinh, sinh);
MAKE_UNARY_TEST(StepFunction, step_function);
MAKE_UNARY_TEST(Square, square);
MAKE_UNARY_TEST(Sqrt, sqrt);
MAKE_UNARY_TEST(Tan, tan);
MAKE_UNARY_TEST(Tanh, tanh);

// test for UnaryPow needs to be handled separately due to the compile-time
// exponent parameter
template <int N>
struct TestFuncEvalUnaryPow {
  template <typename T, typename DistT, typename UniformGen>
  void operator()(UniformGen gen, UniformCustomDistribution<DistT> dist,
                  T /*meta*/) noexcept {
    auto val = make_with_random_values<typename T::type>(make_not_null(&gen),
                                                         make_not_null(&dist));
    CHECK(UnaryPow<N>{}(val) == pow<N>(val)); /*NOLINT*/
  }
};

MAKE_BINARY_TEST(Atan2, atan2);
MAKE_BINARY_TEST(Hypot, hypot);
MAKE_BINARY_TEST(Max, max);
MAKE_BINARY_TEST(Min, min);
MAKE_BINARY_TEST(Pow, pow);

MAKE_BINARY_OP_TEST(Divides, /);
MAKE_BINARY_OP_TEST(Minus, -);
MAKE_BINARY_OP_TEST(Multiplies, *);
MAKE_BINARY_OP_TEST(Plus, +);

MAKE_BINARY_INPLACE_TEST(DivAssign, /=, /);
MAKE_BINARY_INPLACE_TEST(MinusAssign, -=, -);
MAKE_BINARY_INPLACE_TEST(MultAssign, *=, *);
MAKE_BINARY_INPLACE_TEST(PlusAssign, +=, +);

using RealTypeList = tmpl::list<float, double, int, long>;
using AllTypeList = tmpl::list<float, double, std::complex<double>>;
using DoubleSet = tmpl::list<double, std::complex<double>>;

using Bound = std::array<double, 2>;

template <typename ValType, typename Gen, typename Dist>
ValType expandable_random_val(Gen& gen, Dist& dist, size_t /*meta*/) noexcept {
  return make_with_random_values<ValType>(make_not_null(&gen),
                                          make_not_null(&dist));
}

template <typename C, typename ValType, typename Func, typename Gen,
          size_t... Is>
void test_functional_against_function(
    const Func func, Gen& gen, const Bound& bounds,
    const std::index_sequence<Is...> /*meta*/) noexcept {
  static_assert(sizeof...(Is) == C::arity,
                "test was passed incorrect number of arguments");
  UniformCustomDistribution<typename tt::get_fundamental_type_t<ValType>> dist{
      bounds};
  [&func](auto... arg) noexcept {
    if (cpp17::is_same_v<ValType,
                         typename tt::get_fundamental_type_t<ValType>>) {
      CHECK_ITERABLE_APPROX(C{}(arg...), func(arg...));
    } else {
      CHECK(C{}(arg...) == func(arg...));
    }
  }(expandable_random_val<ValType>(gen, dist, Is)...);
}

/// [using_sinusoid]
using Sinusoid = funcl::Multiplies<
    funcl::Identity,
    funcl::Sin<funcl::Plus<funcl::Identity, funcl::Multiplies<>>>>;
///[using_sinusoid]

/// [using_gaussian]
using GaussianExp =
    funcl::Negate<funcl::Square<funcl::Minus<Identity, Literal<5, double>>>>;
using Gaussian = funcl::Exp<GaussianExp>;
/// [using_gaussian]

void test_get_argument() noexcept {
  CHECK(GetArgument<1>{}(-2) == -2);
  CHECK(GetArgument<1, 0>{}(-2) == -2);
  CHECK(GetArgument<2, 0>{}(-2, 1) == -2);
  CHECK(GetArgument<2, 1>{}(-2, 8) == 8);
  CHECK(GetArgument<3, 0>{}(-2, 8, -10) == -2);
  CHECK(GetArgument<3, 1>{}(-2, 8, -10) == 8);
  CHECK(GetArgument<3, 2>{}(-2, 8, -10) == -10);
}

void test_assert_equal() noexcept { CHECK(AssertEqual<>{}(7, 7) == 7); }

template <typename Gen>
void test_functional_combinations(Gen& gen) noexcept {
  const Bound generic{{-50.0, 50.0}};
  const Bound small{{-5.0, 5.0}};

  test_functional_against_function<Plus<Minus<>, Identity>, double>(
      [](const auto& x, const auto& y, const auto& z) { return (x - y) + z; },
      gen, generic, std::make_index_sequence<3>());
  test_functional_against_function<Sinusoid, double>(
      [](const auto& w, const auto& x, const auto& y, const auto& z) {
        return w * sin(x + y * z);
      },
      gen, generic, std::make_index_sequence<4>());
  // This function needs the distribution to be `small` to avoid rare
  // accumulation of error
  test_functional_against_function<Multiplies<Plus<Plus<>, Plus<>>, Identity>,
                                   double>(
      [](const auto& x1, const auto& x2, const auto& x3, const auto& x4,
         const auto& x5) { return x5 * (x1 + x2 + x3 + x4); },
      gen, small, std::make_index_sequence<5>());
  test_functional_against_function<
      Minus<Plus<Identity, Multiplies<>>, Identity>, double>(
      [](const auto& x1, const auto& x2, const auto& x3, const auto& x4) {
        return (x1 + (x2 * x3)) - x4;
      },
      gen, generic, std::make_index_sequence<4>());
  test_functional_against_function<Plus<Negate<>, Identity>, double>(
      [](const auto& x, const auto& y) { return (y - x); }, gen, generic,
      std::make_index_sequence<2>());
  test_functional_against_function<Negate<Plus<>>, double>(
      [](const auto& x, const auto& y) { return -(y + x); }, gen, generic,
      std::make_index_sequence<2>());
  test_functional_against_function<Negate<Plus<Identity, Negate<>>>, double>(
      [](const auto& x, const auto& y) { return -(x - y); }, gen, generic,
      std::make_index_sequence<2>());
  test_functional_against_function<Gaussian, double>(
      [](const auto& x) { return exp(-(x - 5.0) * (x - 5.0)); }, gen, generic,
      std::make_index_sequence<1>());
  test_functional_against_function<Negate<Plus<Literal<3>, Negate<>>>, double>(
      [](const auto& y) { return -(3.0 - y); }, gen, generic,
      std::make_index_sequence<1>());
}

template <typename Gen>
void test_generic_unaries(Gen& gen) noexcept {
  const Bound generic{{-50.0, 50.0}};
  const Bound gt_one{{1.0, 100.0}};
  const Bound mone_one{{-1.0, 1.0}};
  const Bound positive{{.001, 100.0}};

  auto generic_unaries =
      std::make_tuple(std::make_tuple(TestFuncEvalAbs{}, generic),
                      std::make_tuple(TestFuncEvalAcos{}, mone_one),
                      std::make_tuple(TestFuncEvalAcosh{}, gt_one),
                      std::make_tuple(TestFuncEvalAsin{}, mone_one),
                      std::make_tuple(TestFuncEvalAsinh{}, generic),
                      std::make_tuple(TestFuncEvalAtan{}, generic),
                      std::make_tuple(TestFuncEvalAtanh{}, mone_one),
                      std::make_tuple(TestFuncEvalCos{}, generic),
                      std::make_tuple(TestFuncEvalCosh{}, generic),
                      std::make_tuple(TestFuncEvalExp{}, generic),
                      std::make_tuple(TestFuncEvalImag{}, generic),
                      std::make_tuple(TestFuncEvalInvSqrt{}, positive),
                      std::make_tuple(TestFuncEvalLog{}, positive),
                      std::make_tuple(TestFuncEvalReal{}, generic),
                      std::make_tuple(TestFuncEvalSin{}, generic),
                      std::make_tuple(TestFuncEvalSinh{}, generic),
                      std::make_tuple(TestFuncEvalSqrt{}, positive),
                      std::make_tuple(TestFuncEvalSquare{}, generic),
                      std::make_tuple(TestFuncEvalTan{}, generic),
                      std::make_tuple(TestFuncEvalTanh{}, generic),
                      std::make_tuple(TestFuncEvalUnaryPow<1>{}, generic),
                      std::make_tuple(TestFuncEvalUnaryPow<-2>{}, generic),
                      std::make_tuple(TestFuncEvalUnaryPow<3>{}, generic));

  tmpl::for_each<AllTypeList>([&gen, &generic_unaries ](auto x) noexcept {
    using DistType =
        typename tt::get_fundamental_type_t<typename decltype(x)::type>;
    tmpl::for_each<
        tmpl::make_sequence<std::integral_constant<size_t, 0>,
                            std::tuple_size<decltype(generic_unaries)>::value,
                            tmpl::next<tmpl::_1>>>([
      &gen, &x, &generic_unaries
    ](auto index) noexcept {
      auto& tup = std::get<decltype(index)::type::value>(generic_unaries);
      std::get<0>(tup)(
          gen, UniformCustomDistribution<DistType>{std::get<Bound>(tup)}, x);
    });
  });
}

template <typename Gen>
void test_floating_point_functions(Gen& gen) noexcept {
  const Bound generic{{-50.0, 50.0}};
  const Bound gt_one{{1.0, 100.0}};
  const Bound positive{{.001, 100.0}};
  const Bound small{{0.1, 5.0}};

  auto floating_unaries =
      std::make_tuple(std::make_tuple(TestFuncEvalLog10{}, positive));

  auto just_floating_binaries =
      std::make_tuple(std::make_tuple(TestFuncEvalPow{}, small));

  auto generic_binaries =
      std::make_tuple(std::make_tuple(TestFuncEvalDivides{}, gt_one),
                      std::make_tuple(TestFuncEvalMinus{}, generic),
                      std::make_tuple(TestFuncEvalMultiplies{}, generic),
                      std::make_tuple(TestFuncEvalPlus{}, generic),
                      std::make_tuple(TestFuncEvalDivAssign{}, gt_one),
                      std::make_tuple(TestFuncEvalMultAssign{}, generic),
                      std::make_tuple(TestFuncEvalMinusAssign{}, generic),
                      std::make_tuple(TestFuncEvalPlusAssign{}, generic));

  auto floating_binaries =
      std::tuple_cat(generic_binaries, just_floating_binaries);

  tmpl::for_each<DoubleSet>([&gen, &floating_unaries,
                             &floating_binaries ](auto x) noexcept {
    using DistType1 =
        typename tt::get_fundamental_type_t<typename decltype(x)::type>;
    tmpl::for_each<
        tmpl::make_sequence<std::integral_constant<size_t, 0>,
                            std::tuple_size<decltype(floating_unaries)>::value,
                            tmpl::next<tmpl::_1>>>([
      &gen, &x, &floating_unaries
    ](auto index) noexcept {
      auto& tup = std::get<decltype(index)::type::value>(floating_unaries);
      std::get<0>(tup)(
          gen, UniformCustomDistribution<DistType1>{std::get<Bound>(tup)}, x);
    });
    tmpl::for_each<DoubleSet>([&gen, &floating_binaries, &x ](auto y) noexcept {
      using DistType2 =
          typename tt::get_fundamental_type_t<typename decltype(y)::type>;
      tmpl::for_each<tmpl::make_sequence<
          std::integral_constant<size_t, 0>,
          std::tuple_size<decltype(floating_binaries)>::value,
          tmpl::next<tmpl::_1>>>([&gen, &x, &y,
                                  &floating_binaries ](auto index) noexcept {
        auto& tup = std::get<decltype(index)::type::value>(floating_binaries);
        std::get<0>(tup)(
            gen, UniformCustomDistribution<DistType1>{std::get<Bound>(tup)},
            UniformCustomDistribution<DistType2>{std::get<Bound>(tup)}, x, y);
      });
    });
  });
}

template <typename Gen>
void test_real_functions(Gen& gen) noexcept {
  const Bound generic{{-50.0, 50.0}};
  const Bound gt_one{{1.0, 100.0}};
  const Bound positive{{.001, 100.0}};
  const Bound small{{0.1, 5.0}};

  auto generic_binaries =
      std::make_tuple(std::make_tuple(TestFuncEvalDivides{}, gt_one),
                      std::make_tuple(TestFuncEvalMinus{}, generic),
                      std::make_tuple(TestFuncEvalMultiplies{}, generic),
                      std::make_tuple(TestFuncEvalPlus{}, generic),
                      std::make_tuple(TestFuncEvalDivAssign{}, gt_one),
                      std::make_tuple(TestFuncEvalMultAssign{}, generic),
                      std::make_tuple(TestFuncEvalMinusAssign{}, generic),
                      std::make_tuple(TestFuncEvalPlusAssign{}, generic));

  auto just_real_binaries =
      std::make_tuple(std::make_tuple(TestFuncEvalAtan2{}, generic),
                      std::make_tuple(TestFuncEvalHypot{}, generic));

  auto same_just_real_binaries =
      std::make_tuple(std::make_tuple(TestFuncEvalMax{}, generic),
                      std::make_tuple(TestFuncEvalMin{}, generic));

  auto real_unaries =
      std::make_tuple(std::make_tuple(TestFuncEvalCbrt{}, positive),
                      std::make_tuple(TestFuncEvalErf{}, generic),
                      std::make_tuple(TestFuncEvalExp2{}, small),
                      std::make_tuple(TestFuncEvalFabs{}, generic),
                      std::make_tuple(TestFuncEvalInvCbrt{}, gt_one),
                      std::make_tuple(TestFuncEvalLog2{}, gt_one),
                      std::make_tuple(TestFuncEvalStepFunction{}, generic));

  auto real_binaries = std::tuple_cat(just_real_binaries, generic_binaries);

  tmpl::for_each<RealTypeList>([
    &gen, &real_binaries, &same_just_real_binaries, &real_unaries
  ](auto x) noexcept {
    using DistType1 =
        typename tt::get_fundamental_type_t<typename decltype(x)::type>;
    tmpl::for_each<tmpl::make_sequence<
        std::integral_constant<size_t, 0>,
        std::tuple_size<decltype(real_unaries)>::value, tmpl::next<tmpl::_1>>>(
        [&gen, &x, &real_unaries ](auto index) noexcept {
          auto& tup = std::get<decltype(index)::type::value>(real_unaries);
          std::get<0>(tup)(
              gen, UniformCustomDistribution<DistType1>{std::get<Bound>(tup)},
              x);
        });

    tmpl::for_each<tmpl::make_sequence<
        std::integral_constant<size_t, 0>,
        std::tuple_size<decltype(same_just_real_binaries)>::value,
        tmpl::next<tmpl::_1>>>(
        [&gen, &x, &same_just_real_binaries ](auto index) noexcept {
          auto& tup =
              std::get<decltype(index)::type::value>(same_just_real_binaries);
          std::get<0>(tup)(
              gen, UniformCustomDistribution<DistType1>{std::get<Bound>(tup)},
              UniformCustomDistribution<DistType1>{std::get<Bound>(tup)}, x, x);
        });

    tmpl::for_each<RealTypeList>([&gen, &real_binaries, &x ](auto y) noexcept {
      using DistType2 =
          typename tt::get_fundamental_type_t<typename decltype(y)::type>;
      tmpl::for_each<
          tmpl::make_sequence<std::integral_constant<size_t, 0>,
                              std::tuple_size<decltype(real_binaries)>::value,
                              tmpl::next<tmpl::_1>>>([
        &gen, &x, &y, &real_binaries
      ](auto index) noexcept {
        auto& tup = std::get<decltype(index)::type::value>(real_binaries);
        std::get<0>(tup)(
            gen, UniformCustomDistribution<DistType1>(std::get<Bound>(tup)),
            UniformCustomDistribution<DistType2>(std::get<Bound>(tup)), x, y);
      });
    });
  });
}

SPECTRE_TEST_CASE("Unit.Utilities.Functional", "[Utilities][Unit]") {
  MAKE_GENERATOR(generator);
  test_generic_unaries(generator);
  test_floating_point_functions(generator);
  test_real_functions(generator);
  test_functional_combinations(generator);
  test_assert_equal();
  test_get_argument();
}

// [[OutputRegex, Values are not equal in funcl::AssertEqual 7 and 8]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.Functional.AssertEqual",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  AssertEqual<>{}(7, 8);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
#undef MAKE_UNARY_TEST
#undef MAKE_BINARY_TEST
#undef MAKE_BINARY_OP_TEST
#undef MAKE_BINARY_INPLACE_TEST
}  // namespace
}  // namespace funcl
