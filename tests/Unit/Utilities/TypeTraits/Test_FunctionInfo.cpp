// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/FunctionInfo.hpp"

namespace {
// The name syntax is ReturnType_NumberOfArgs_Counter
// clang-tidy: no non-const references (we need them for testing)
void void_none() {}
double double_none() { return 1.0; }
void void_one_0(double /*unused*/) {}
void void_one_1(const double& /*unused*/) {}
void void_one_2(double& /*unused*/) {}  // NOLINT
void void_one_3(double* /*unused*/) {}
void void_one_4(const double* /*unused*/) {}
void void_one_5(const double* const /*unused*/) {}

int int_one_0(double /*unused*/) { return 0; }
int int_one_1(const double& /*unused*/) { return 0; }  // NOLINT
int int_one_2(double& /*unused*/) { return 0; }        // NOLINT
int int_one_3(double* /*unused*/) { return 0; }
int int_one_4(const double* /*unused*/) { return 0; }
int int_one_5(const double* const /*unused*/) { return 0; }

int int_two_0(double /*unused*/, char* /*unused*/) { return 0; }
int int_two_1(const double& /*unused*/, const char* /*unused*/) { return 0; }
int int_two_2(double& /*unused*/, const char* const /*unused*/) {  // NOLINT
  return 0;
}
int int_two_3(double* /*unused*/, char& /*unused*/) { return 0; }  // NOLINT
int int_two_4(const double* /*unused*/, const char& /*unused*/) { return 0; }
int int_two_5(const double* const /*unused*/, char /*unused*/) { return 0; }

// We have to NOLINT these for 2 reasons:
// - no non-const references
// - ClangTidy wants a () after the macros, which expands to not what we want
#define FUNCTION_INFO_TEST(NAME, PREFIX, POSTFIX)                              \
  struct FunctionInfoTest##NAME {                                              \
    PREFIX void void_none() POSTFIX {}                            /* NOLINT */ \
    PREFIX double double_none() POSTFIX { return 1.0; }           /* NOLINT */ \
    PREFIX void void_one_0(double /*unused*/) POSTFIX {}          /* NOLINT */ \
    PREFIX void void_one_1(const double& /*unused*/) POSTFIX {}   /* NOLINT */ \
    PREFIX void void_one_2(double& /*unused*/) POSTFIX {}         /* NOLINT */ \
    PREFIX void void_one_3(double* /*unused*/) POSTFIX {}         /* NOLINT */ \
    PREFIX void void_one_4(const double* /*unused*/) POSTFIX {}   /* NOLINT */ \
    PREFIX void void_one_5(                                       /* NOLINT */ \
                           const double* const /*unused*/)        /* NOLINT */ \
        POSTFIX {}                                                /* NOLINT */ \
    PREFIX int int_one_0(double /*unused*/) POSTFIX { return 0; } /* NOLINT */ \
    PREFIX int int_one_1(const double& /*unused*/) POSTFIX {      /* NOLINT */ \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_2(double& /*unused*/) POSTFIX { /* NOLINT */            \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_3(double* /*unused*/) POSTFIX { /* NOLINT */            \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_4(const double* /*unused*/) POSTFIX { /* NOLINT */      \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_one_5(const double* const /*unused*/) /* NOLINT */          \
        POSTFIX {                                        /* NOLINT */          \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_0(double /*unused*/,          /* NOLINT */              \
                         char* /*unused*/) POSTFIX { /* NOLINT */              \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_1(const double& /*unused*/,         /* NOLINT */        \
                         const char* /*unused*/) POSTFIX { /* NOLINT */        \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_2(double& /*unused*/,                     /* NOLINT */  \
                         const char* const /*unused*/) POSTFIX { /* NOLINT */  \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_3(double* /*unused*/,         /* NOLINT */              \
                         char& /*unused*/) POSTFIX { /* NOLINT */              \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_4(const double* /*unused*/,         /* NOLINT */        \
                         const char& /*unused*/) POSTFIX { /* NOLINT */        \
      return 0;                                                                \
    }                                                                          \
    PREFIX int int_two_5(const double* const /*unused*/, /* NOLINT */          \
                         char /*unused*/) POSTFIX {      /* NOLINT */          \
      return 0;                                                                \
    }                                                                          \
  };

FUNCTION_INFO_TEST(, , )
FUNCTION_INFO_TEST(Const, , const)
FUNCTION_INFO_TEST(Static, static, )
FUNCTION_INFO_TEST(Noexcept, , )
FUNCTION_INFO_TEST(ConstNoexcept, , const)
FUNCTION_INFO_TEST(StaticNoexcept, static, )

#undef FUNCTION_INFO_TEST

struct FunctionInfoTestVirtual {
  FunctionInfoTestVirtual(const FunctionInfoTestVirtual&) = default;
  FunctionInfoTestVirtual& operator=(const FunctionInfoTestVirtual&) = default;
  FunctionInfoTestVirtual(FunctionInfoTestVirtual&&) = default;
  FunctionInfoTestVirtual& operator=(FunctionInfoTestVirtual&&) = default;
  virtual ~FunctionInfoTestVirtual() = default;
  virtual void void_none() {}
  virtual double double_none() { return 1.0; }
  virtual void void_one_0(double /*unused*/) {}
  virtual void void_one_1(const double& /*unused*/) {}
  virtual void void_one_2(double& /*unused*/) {}  // NOLINT
  virtual void void_one_3(double* /*unused*/) {}
  virtual void void_one_4(const double* /*unused*/) {}
  virtual void void_one_5(const double* const /*unused*/) {}

  virtual int int_one_0(double /*unused*/) { return 0; }
  virtual int int_one_1(const double& /*unused*/) { return 0; }
  virtual int int_one_2(double& /*unused*/) { return 0; }  // NOLINT
  virtual int int_one_3(double* /*unused*/) { return 0; }
  virtual int int_one_4(const double* /*unused*/) { return 0; }
  virtual int int_one_5(const double* const /*unused*/) { return 0; }

  virtual int int_two_0(double /*unused*/, char* /*unused*/) { return 0; }
  virtual int int_two_1(const double& /*unused*/,  // NOLINT
                        const char* /*unused*/) {
    return 0;
  }
  virtual int int_two_2(double& /*unused*/,  // NOLINT
                        const char* const /*unused*/) {
    return 0;
  }
  virtual int int_two_3(double* /*unused*/, char& /*unused*/) {  // NOLINT
    return 0;
  }
  virtual int int_two_4(const double* /*unused*/, const char& /*unused*/) {
    return 0;
  }
  virtual int int_two_5(const double* const /*unused*/, char /*unused*/) {
    return 0;
  }
};

struct FunctionInfoTestVirtualBase {
  // Some of these are redundant because we don't need to const things in
  // forward decls. However, so that we can use the generic interface we keep
  // the functions.
  FunctionInfoTestVirtualBase(const FunctionInfoTestVirtualBase&) = default;
  FunctionInfoTestVirtualBase& operator=(const FunctionInfoTestVirtualBase&) =
      default;
  FunctionInfoTestVirtualBase(FunctionInfoTestVirtualBase&&) = default;
  FunctionInfoTestVirtualBase& operator=(FunctionInfoTestVirtualBase&&) =
      default;
  virtual ~FunctionInfoTestVirtualBase() = default;
  virtual void void_none() = 0;
  virtual double double_none() = 0;
  virtual void void_one_0(double /*unused*/) = 0;
  virtual void void_one_1(const double& /*unused*/) = 0;
  virtual void void_one_2(double& /*unused*/) = 0;  // NOLINT
  virtual void void_one_3(double* /*unused*/) = 0;
  virtual void void_one_4(const double* /*unused*/) = 0;
  virtual void void_one_5(const double* /*unused*/) = 0;

  virtual int int_one_0(double /*unused*/) = 0;
  virtual int int_one_1(const double& /*unused*/) = 0;
  virtual int int_one_2(double& /*unused*/) = 0;  // NOLINT
  virtual int int_one_3(double* /*unused*/) = 0;
  virtual int int_one_4(const double* /*unused*/) = 0;
  virtual int int_one_5(const double* /*unused*/) = 0;

  virtual int int_two_0(double /*unused*/, char* /*unused*/) = 0;
  virtual int int_two_1(const double& /*unused*/,                   // NOLINT
                        const char* /*unused*/) = 0;                // NOLINT
  virtual int int_two_2(double& /*unused*/,                         // NOLINT
                        const char* /*unused*/) = 0;                // NOLINT
  virtual int int_two_3(double* /*unused*/, char& /*unused*/) = 0;  // NOLINT
  virtual int int_two_4(const double* /*unused*/, const char& /*unused*/) = 0;
  virtual int int_two_5(const double* /*unused*/, char /*unused*/) = 0;
};

// Check that function_info works after LazyF is applied to a variety of
// different functions. LazyF is identity, add_pointer, add_const<add_pointer>,
// add_volatile<add_pointer>, and add_cv<add_pointer>. This allows us to check
// all the different combinations with a single class.
template <template <class> class LazyF>
struct check_function_info {
  template <class T>
  using F = typename LazyF<T>::type;

  template <class Function, class ReturnType, class ClassType, class... Args>
  struct Check {
    static_assert(
        std::is_same_v<ReturnType,
                       typename tt::function_info<Function>::return_type>,
        "Failed testing function_info");
    static_assert(
        std::is_same_v<tmpl::list<Args...>,
                       typename tt::function_info<Function>::argument_types>,
        "Failed testing function_info");
    static_assert(
        std::is_same_v<ClassType,
                       typename tt::function_info<Function>::class_type>,
        "Failed testing function_info");
    static constexpr bool t = true;
  };

  static constexpr Check<F<decltype(void_none)>, void, void> t_void_none{};
  static constexpr Check<F<decltype(double_none)>, double, void>
      t_double_none{};

  static constexpr Check<F<decltype(void_one_0)>, void, void, double>
      t_void_one_0{};
  static constexpr Check<F<decltype(void_one_1)>, void, void, const double&>
      t_void_one_1{};
  static constexpr Check<F<decltype(void_one_2)>, void, void, double&>
      t_void_one_2{};
  static constexpr Check<F<decltype(void_one_3)>, void, void, double*>
      t_void_one_3{};
  static constexpr Check<F<decltype(void_one_4)>, void, void, const double*>
      t_void_one_4{};
  static constexpr Check<F<decltype(void_one_5)>, void, void, const double*>
      t_void_one_5{};

  static constexpr Check<F<decltype(int_one_0)>, int, void, double>
      t_int_one_0{};
  static constexpr Check<F<decltype(int_one_1)>, int, void, const double&>
      t_int_one_1{};
  static constexpr Check<F<decltype(int_one_2)>, int, void, double&>
      t_int_one_2{};
  static constexpr Check<F<decltype(int_one_3)>, int, void, double*>
      t_int_one_3{};
  static constexpr Check<F<decltype(int_one_4)>, int, void, const double*>
      t_int_one_4{};
  static constexpr Check<F<decltype(int_one_5)>, int, void, const double*>
      t_int_one_5{};

  static constexpr Check<F<decltype(int_two_0)>, int, void, double, char*>
      t_int_two_0{};
  static constexpr Check<F<decltype(int_two_1)>, int, void, const double&,
                         const char*>
      t_int_two_1{};
  static constexpr Check<F<decltype(int_two_2)>, int, void, double&,
                         const char*>
      t_int_two_2{};
  static constexpr Check<F<decltype(int_two_3)>, int, void, double*, char&>
      t_int_two_3{};
  static constexpr Check<F<decltype(int_two_4)>, int, void, const double*,
                         const char&>
      t_int_two_4{};
  static constexpr Check<F<decltype(int_two_5)>, int, void, const double*, char>
      t_int_two_5{};

  template <class Scope, class Class = Scope>
  struct CheckClass {
    // We have to remove the pointer since we are already adding a pointer
    // sometimes using F
    static constexpr bool t_void_none =
        Check<std::remove_pointer_t<F<decltype(&Scope::void_none)>>, void,
              Class>::t;
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::double_none)>>, double, Class>
        t_double_none{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_0)>>, void, Class,
        double>
        t_void_one_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_1)>>, void, Class,
        const double&>
        t_void_one_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_2)>>, void, Class,
        double&>
        t_void_one_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_3)>>, void, Class,
        double*>
        t_void_one_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_4)>>, void, Class,
        const double*>
        t_void_one_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::void_one_5)>>, void, Class,
        const double*>
        t_void_one_5{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_0)>>, int, Class,
        double>
        t_int_one_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_1)>>, int, Class,
        const double&>
        t_int_one_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_2)>>, int, Class,
        double&>
        t_int_one_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_3)>>, int, Class,
        double*>
        t_int_one_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_4)>>, int, Class,
        const double*>
        t_int_one_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_one_5)>>, int, Class,
        const double*>
        t_int_one_5{};

    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_0)>>, int, Class,
        double, char*>
        t_int_two_0{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_1)>>, int, Class,
        const double&, const char*>
        t_int_two_1{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_2)>>, int, Class,
        double&, const char*>
        t_int_two_2{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_3)>>, int, Class,
        double*, char&>
        t_int_two_3{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_4)>>, int, Class,
        const double*, const char&>
        t_int_two_4{};
    static constexpr Check<
        std::remove_pointer_t<F<decltype(&Scope::int_two_5)>>, int, Class,
        const double*, char>
        t_int_two_5{};
  };

  static constexpr CheckClass<FunctionInfoTest> non_const_members{};
  static constexpr CheckClass<FunctionInfoTestConst> const_members{};
  static constexpr CheckClass<FunctionInfoTestConstNoexcept>
      const_noexcept_members{};
  static constexpr CheckClass<FunctionInfoTestNoexcept> _members{};
  static constexpr CheckClass<FunctionInfoTestVirtual> virtual_members{};
  static constexpr CheckClass<FunctionInfoTestVirtualBase>
      virtual_base_members{};
  static constexpr CheckClass<FunctionInfoTestStatic, void> static_members{};
  static constexpr CheckClass<FunctionInfoTestStaticNoexcept, void>
      static_noexcept_members{};
};

template <template <class> class F>
struct add_pointer_helper {
  template <class T>
  struct impl {
    using type = typename F<std::add_pointer_t<T>>::type;
  };
};

template <class T>
struct identity {
  using type = T;
};

// Scope function calls to avoid warnings
struct TestFunctions {
  TestFunctions() {
    (void)check_function_info<identity>{};
    (void)check_function_info<std::add_pointer>{};
    (void)check_function_info<add_pointer_helper<std::add_const>::impl>{};
    (void)check_function_info<add_pointer_helper<std::add_volatile>::impl>{};
    (void)check_function_info<add_pointer_helper<std::add_cv>::impl>{};
    // Use these to avoid warnings
    void_none();
    (void)double_none();

    double t_double = 1.0;
    void_one_0(t_double);
    void_one_1(t_double);
    void_one_2(t_double);
    void_one_3(&t_double);
    void_one_4(&t_double);
    void_one_5(&t_double);

    (void)int_one_0(t_double);
    (void)int_one_1(t_double);
    (void)int_one_2(t_double);
    (void)int_one_3(&t_double);
    (void)int_one_4(&t_double);
    (void)int_one_5(&t_double);

    char t_char = 'a';
    (void)int_two_0(t_double, &t_char);
    (void)int_two_1(t_double, &t_char);
    (void)int_two_2(t_double, &t_char);
    (void)int_two_3(&t_double, t_char);
    (void)int_two_4(&t_double, t_char);
    (void)int_two_5(&t_double, t_char);
  }
};
TestFunctions test_functions{};
}  // namespace
