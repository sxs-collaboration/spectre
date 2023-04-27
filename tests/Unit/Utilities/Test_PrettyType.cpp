// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <deque>
#include <forward_list>
#include <istream>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.PrettyType.Fundamental",
                  "[Utilities][Unit]") {
  CHECK("char" == pretty_type::get_name<char>());
  CHECK("signed char" == pretty_type::get_name<signed char>());
  CHECK("unsigned char" == pretty_type::get_name<unsigned char>());
  CHECK("wchar_t" == pretty_type::get_name<wchar_t>());
  CHECK("char16_t" == pretty_type::get_name<char16_t>());
  CHECK("char32_t" == pretty_type::get_name<char32_t>());
  CHECK("int" == pretty_type::get_name<int>());
  CHECK("unsigned int" == pretty_type::get_name<unsigned int>());
  CHECK("long" == pretty_type::get_name<long>());
  CHECK("unsigned long" == pretty_type::get_name<unsigned long>());
  CHECK("long long" == pretty_type::get_name<long long>());
  CHECK("unsigned long long" == pretty_type::get_name<unsigned long long>());
  CHECK("short" == pretty_type::get_name<short>());
  CHECK("unsigned short" == pretty_type::get_name<unsigned short>());
  CHECK("float" == pretty_type::get_name<float>());
  CHECK("double" == pretty_type::get_name<double>());
  CHECK("long double" == pretty_type::get_name<long double>());
  CHECK("bool" == pretty_type::get_name<bool>());

  // References and pointers
  CHECK("double&" == pretty_type::get_name<double&>());
  CHECK("double const&" == pretty_type::get_name<const double&>());
  CHECK("double volatile&" == pretty_type::get_name<volatile double&>());
  CHECK("double const volatile&" ==
        pretty_type::get_name<const volatile double&>());
  CHECK("double*" == pretty_type::get_name<double*>());
  CHECK("double* const" == pretty_type::get_name<double* const>());
  CHECK("double* volatile" == pretty_type::get_name<double* volatile>());
  CHECK("double* const volatile" ==
        pretty_type::get_name<double* volatile const>());
  CHECK("double const* const" == pretty_type::get_name<const double* const>());
  CHECK("double volatile* volatile" ==
        pretty_type::get_name<volatile double* volatile>());
  CHECK("double volatile* const" ==
        pretty_type::get_name<volatile double* const>());
  CHECK("double const* volatile" ==
        pretty_type::get_name<const double* volatile>());
  CHECK("double volatile* const volatile" ==
        pretty_type::get_name<volatile double* const volatile>());
  CHECK("double const volatile* volatile" ==
        pretty_type::get_name<const volatile double* volatile>());
  CHECK("double const volatile* const volatile" ==
        pretty_type::get_name<const volatile double* const volatile>());
  CHECK("double const*" == pretty_type::get_name<const double*>());
  CHECK("double volatile*" == pretty_type::get_name<volatile double*>());
  CHECK("double const volatile*" ==
        pretty_type::get_name<const volatile double*>());

  // Test get_runtime_type_name
  CHECK("char" == pretty_type::get_runtime_type_name('a'));
}

SPECTRE_TEST_CASE("Unit.Utilities.PrettyType.Stl", "[Utilities][Unit]") {
  CHECK("std::string" == pretty_type::get_name<std::string>());
  CHECK("std::array<double, 4>" ==
        (pretty_type::get_name<std::array<double, 4>>()));
  CHECK("std::vector<double>" == pretty_type::get_name<std::vector<double>>());
  CHECK("std::deque<double>" == pretty_type::get_name<std::deque<double>>());
  CHECK("std::forward_list<double>" ==
        pretty_type::get_name<std::forward_list<double>>());
  CHECK("std::list<double>" == pretty_type::get_name<std::list<double>>());
  CHECK("std::map<std::string, double>" ==
        (pretty_type::get_name<std::map<std::string, double>>()));
  CHECK("std::set<std::string>" ==
        (pretty_type::get_name<std::set<std::string>>()));
  CHECK("std::multiset<std::string>" ==
        (pretty_type::get_name<std::multiset<std::string>>()));
  CHECK("std::multimap<std::string, double>" ==
        (pretty_type::get_name<std::multimap<std::string, double>>()));
  CHECK("std::unordered_map<std::string, double>" ==
        (pretty_type::get_name<std::unordered_map<std::string, double>>()));
  CHECK(
      "std::unordered_multimap<std::string, double>" ==
      (pretty_type::get_name<std::unordered_multimap<std::string, double>>()));
  CHECK("std::unordered_set<std::string>" ==
        (pretty_type::get_name<std::unordered_set<std::string>>()));
  CHECK("std::unordered_multiset<std::string>" ==
        (pretty_type::get_name<std::unordered_multiset<std::string>>()));

  CHECK("std::priority_queue<double, std::vector<double>>" ==
        pretty_type::get_name<std::priority_queue<double>>());
  CHECK("std::queue<double, std::deque<double>>" ==
        pretty_type::get_name<std::queue<double>>());
  CHECK("std::stack<double, std::deque<double>>" ==
        pretty_type::get_name<std::stack<double>>());

  CHECK("std::unique_ptr<double>" ==
        pretty_type::get_name<std::unique_ptr<double>>());
  CHECK("std::shared_ptr<double>" ==
        pretty_type::get_name<std::shared_ptr<double>>());
  CHECK("std::weak_ptr<double>" ==
        pretty_type::get_name<std::weak_ptr<double>>());
}

// NOTE: Do not put these in a namespace, including an anonymous
// namespace.  These are for testing demangling things that are not in
// a namespace.
struct Test_PrettyType_struct {};
template <typename>
struct Test_PrettyType_templated_struct {};
enum class Test_PrettyType_Enum { Zero, One, Two };

namespace Test_PrettyType_namespace {
struct NamedNamespace {};
}  // namespace Test_PrettyType_namespace

namespace {
struct Type1Containing2Digits3 {};

struct TestType {};

template <typename>
struct Template {};

template <typename, typename>
struct Template2 {};

template <typename...>
struct Pack {
  struct Inner {};
};

template <typename, typename...>
struct Pack2 {};

struct Outer {
  struct Inner {};

  template <typename>
  struct InnerTemplate {};
};

template <typename>
struct OuterTemplate {
  struct Inner {};

  template <typename>
  struct InnerTemplate {};
};

template <int>
struct NonType {};

template <int, int>
struct NonTypeNonType {};

template <typename, int>
struct TypeNonType {};

template <int, typename>
struct NonTypeType {};

template <std::nullptr_t>
struct NullptrTemplate {};

enum class LocalEnum { Zero, One, Two, Negative = -1 };

template <Test_PrettyType_Enum>
struct GlobalEnumTemplate {};

template <LocalEnum>
struct LocalEnumTemplate {};

template <typename>
struct TemplateWithName {
  static std::string name() { return "UniqueTemplateWithName"; }
};

struct NonTemplateWithName {
  static std::string name() { return "UniqueNonTemplateWithName"; }
};

struct ShortName {
  template <typename TestType>
  static std::string name() {
    return pretty_type::short_name<TestType>();
  }
};

struct Name {
  template <typename TestType>
  static std::string name() {
    if constexpr (std::is_constructible_v<TestType>) {
      CHECK(pretty_type::name<TestType>() == pretty_type::name(TestType{}));
    }
    return pretty_type::name<TestType>();
  }
};

template <typename NameFunc>
void test_name_func() {
  // Fundamentals
  CHECK(NameFunc::template name<bool>() == "bool");
  CHECK(NameFunc::template name<char>() == "char");
  CHECK(NameFunc::template name<signed char>() == "signed char");
  CHECK(NameFunc::template name<unsigned char>() == "unsigned char");
  CHECK(NameFunc::template name<short>() == "short");
  CHECK(NameFunc::template name<unsigned short>() == "unsigned short");
  CHECK(NameFunc::template name<int>() == "int");
  CHECK(NameFunc::template name<unsigned int>() == "unsigned int");
  CHECK(NameFunc::template name<long>() == "long");
  CHECK(NameFunc::template name<unsigned long>() == "unsigned long");
  CHECK(NameFunc::template name<long long>() == "long long");
  CHECK(NameFunc::template name<unsigned long long>() == "unsigned long long");
  CHECK(NameFunc::template name<void>() == "void");
  CHECK(NameFunc::template name<float>() == "float");
  CHECK(NameFunc::template name<double>() == "double");
  CHECK(NameFunc::template name<long double>() == "long double");

  // Standard library
  // Untemplated
  CHECK(NameFunc::template name<std::type_info>() == "type_info");
  // Templated
  CHECK(NameFunc::template name<std::vector<int>>() == "vector");
  // (Probably) special cased in mangling
  CHECK(NameFunc::template name<std::ostream>() == "ostream");
  // Possibly special cased in mangling and a case we particularly care about
  CHECK(NameFunc::template name<std::string>() == "string");

  // Types and templates with no namespaces
  CHECK(NameFunc::template name<Test_PrettyType_struct>() ==
        "Test_PrettyType_struct");
  CHECK(NameFunc::template name<Test_PrettyType_templated_struct<int>>() ==
        "Test_PrettyType_templated_struct");
  CHECK(NameFunc::template name<
            Test_PrettyType_templated_struct<Test_PrettyType_struct>>() ==
        "Test_PrettyType_templated_struct");
  CHECK(NameFunc::template name<Test_PrettyType_templated_struct<
            Test_PrettyType_templated_struct<int>>>() ==
        "Test_PrettyType_templated_struct");
  CHECK(NameFunc::template name<Test_PrettyType_templated_struct<
            Test_PrettyType_templated_struct<Test_PrettyType_struct>>>() ==
        "Test_PrettyType_templated_struct");

  // Named namespaces
  CHECK(NameFunc::template name<Test_PrettyType_namespace::NamedNamespace>() ==
        "NamedNamespace");

  // Anonymous namespaces
  CHECK(NameFunc::template name<TestType>() == "TestType");

  // Digits (special meaning in mangled names)
  CHECK(NameFunc::template name<Type1Containing2Digits3>() ==
        "Type1Containing2Digits3");

  using Global = Test_PrettyType_struct;
  using Std = std::type_info;
  using Special = std::ostream;
  using Templated = Test_PrettyType_templated_struct<Test_PrettyType_struct>;

  // const
  CHECK(NameFunc::template name<const int>() == "int");
  CHECK(NameFunc::template name<const Global>() == "Test_PrettyType_struct");
  CHECK(NameFunc::template name<const TestType>() == "TestType");
  CHECK(NameFunc::template name<const Std>() == "type_info");
  CHECK(NameFunc::template name<const Special>() == "ostream");
  CHECK(NameFunc::template name<const Templated>() ==
        "Test_PrettyType_templated_struct");

  // Template stuff
  CHECK(NameFunc::template name<Template<int>>() == "Template");
  CHECK(NameFunc::template name<Template<Global>>() == "Template");
  CHECK(NameFunc::template name<Template<TestType>>() == "Template");
  CHECK(NameFunc::template name<Template<Std>>() == "Template");
  CHECK(NameFunc::template name<Template<Special>>() == "Template");
  CHECK(NameFunc::template name<Template<Templated>>() == "Template");
  CHECK(NameFunc::template name<Template<const Global>>() == "Template");
  CHECK(NameFunc::template name<Template<const TestType>>() == "Template");

  CHECK(NameFunc::template name<Template2<int, int>>() == "Template2");
  CHECK(NameFunc::template name<Template2<int, Global>>() == "Template2");
  CHECK(NameFunc::template name<Template2<int, TestType>>() == "Template2");
  CHECK(NameFunc::template name<Template2<int, Std>>() == "Template2");
  CHECK(NameFunc::template name<Template2<int, Special>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Global, int>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Global, Global>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Global, TestType>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Global, Std>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Global, Special>>() == "Template2");
  CHECK(NameFunc::template name<Template2<TestType, int>>() == "Template2");
  CHECK(NameFunc::template name<Template2<TestType, Global>>() == "Template2");
  CHECK(NameFunc::template name<Template2<TestType, TestType>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<TestType, Std>>() == "Template2");
  CHECK(NameFunc::template name<Template2<TestType, Special>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Std, int>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Std, Global>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Std, TestType>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Std, Std>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Std, Special>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Special, int>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Special, Global>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Special, TestType>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Special, Std>>() == "Template2");
  CHECK(NameFunc::template name<Template2<Special, Special>>() == "Template2");

  CHECK(NameFunc::template name<Template2<Global, const Global>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<Global, const TestType>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<const Global, Global>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<const TestType, Global>>() ==
        "Template2");

  CHECK(NameFunc::template name<Template<Template<int>>>() == "Template");
  CHECK(NameFunc::template name<Template2<Template<int>, Template<double>>>() ==
        "Template2");

  CHECK(NameFunc::template name<Pack<>>() == "Pack");
  CHECK(NameFunc::template name<Pack<int>>() == "Pack");
  CHECK(NameFunc::template name<Pack<Global>>() == "Pack");
  CHECK(NameFunc::template name<Pack<TestType>>() == "Pack");
  CHECK(NameFunc::template name<Pack<Std>>() == "Pack");
  CHECK(NameFunc::template name<Pack<Special>>() == "Pack");
  CHECK(NameFunc::template name<Pack<Templated>>() == "Pack");

  CHECK(NameFunc::template name<Pack2<int>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<Global>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<TestType>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<Std>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<Special>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<Templated>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<TestType, TestType>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<Special, TestType>>() == "Pack2");
  CHECK(NameFunc::template name<Pack2<TestType, Special>>() == "Pack2");

  // Nested types
  CHECK(NameFunc::template name<Outer::Inner>() == "Inner");
  CHECK(NameFunc::template name<Outer::InnerTemplate<int>>() ==
        "InnerTemplate");
  CHECK(NameFunc::template name<Outer::InnerTemplate<Global>>() ==
        "InnerTemplate");
  CHECK(NameFunc::template name<Outer::InnerTemplate<TestType>>() ==
        "InnerTemplate");
  CHECK(NameFunc::template name<Outer::InnerTemplate<Special>>() ==
        "InnerTemplate");
  CHECK(NameFunc::template name<OuterTemplate<int>::Inner>() == "Inner");
  CHECK(NameFunc::template name<OuterTemplate<Global>::Inner>() == "Inner");
  CHECK(NameFunc::template name<OuterTemplate<TestType>::Inner>() == "Inner");
  CHECK(NameFunc::template name<OuterTemplate<Special>::Inner>() == "Inner");

  CHECK(NameFunc::template name<Pack<>::Inner>() == "Inner");
  CHECK(NameFunc::template name<Pack<TestType>::Inner>() == "Inner");

  // Non-type template parameters
  CHECK(NameFunc::template name<NonType<3>>() == "NonType");
  CHECK(NameFunc::template name<NonType<0>>() == "NonType");
  CHECK(NameFunc::template name<NonType<-3>>() == "NonType");
  CHECK(NameFunc::template name<NonTypeNonType<3, 3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<3, 0>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<3, -3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<0, 3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<0, 0>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<0, -3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<-3, 3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<-3, 0>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<NonTypeNonType<-3, -3>>() == "NonTypeNonType");
  CHECK(NameFunc::template name<TypeNonType<int, 3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Global, 3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<TestType, 3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Special, 3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<int, 0>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Global, 0>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<TestType, 0>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Special, 0>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<int, -3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Global, -3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<TestType, -3>>() == "TypeNonType");
  CHECK(NameFunc::template name<TypeNonType<Special, -3>>() == "TypeNonType");
  CHECK(NameFunc::template name<NonTypeType<3, int>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<3, Global>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<3, TestType>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<3, Special>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<0, int>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<0, Global>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<0, TestType>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<0, Special>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<-3, int>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<-3, Global>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<-3, TestType>>() == "NonTypeType");
  CHECK(NameFunc::template name<NonTypeType<-3, Special>>() == "NonTypeType");

  // nullptr-related things
  CHECK(NameFunc::template name<Template<std::nullptr_t>>() == "Template");
  CHECK(NameFunc::template name<Template2<std::nullptr_t, std::nullptr_t>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<int, std::nullptr_t>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<Global, std::nullptr_t>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<TestType, std::nullptr_t>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<Special, std::nullptr_t>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<std::nullptr_t, int>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<std::nullptr_t, Global>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<std::nullptr_t, TestType>>() ==
        "Template2");
  CHECK(NameFunc::template name<Template2<std::nullptr_t, Special>>() ==
        "Template2");
  CHECK(NameFunc::template name<NullptrTemplate<nullptr>>() ==
        "NullptrTemplate");

  // Enum non-type template parameters
  CHECK(NameFunc::template name<
            GlobalEnumTemplate<Test_PrettyType_Enum::Zero>>() ==
        "GlobalEnumTemplate");
  CHECK(NameFunc::template name<
            GlobalEnumTemplate<Test_PrettyType_Enum::One>>() ==
        "GlobalEnumTemplate");
  CHECK(NameFunc::template name<
            Template2<GlobalEnumTemplate<Test_PrettyType_Enum::One>,
                      GlobalEnumTemplate<Test_PrettyType_Enum::Two>>>() ==
        "Template2");
  CHECK(NameFunc::template name<LocalEnumTemplate<LocalEnum::Zero>>() ==
        "LocalEnumTemplate");
  CHECK(NameFunc::template name<LocalEnumTemplate<LocalEnum::One>>() ==
        "LocalEnumTemplate");
  CHECK(
      NameFunc::template name<Template2<LocalEnumTemplate<LocalEnum::One>,
                                        LocalEnumTemplate<LocalEnum::Two>>>() ==
      "Template2");
  CHECK(NameFunc::template name<LocalEnumTemplate<LocalEnum::Negative>>() ==
        "LocalEnumTemplate");

  // Complicated Test
  CHECK(NameFunc::template name<OuterTemplate<
            Template2<OuterTemplate<Global>::InnerTemplate<TestType>,
                      OuterTemplate<Global>::InnerTemplate<TestType>>>::
                                    InnerTemplate<NonTypeType<1, Global>>>() ==
        "InnerTemplate");

  // Long test
  CHECK(NameFunc::template name<tmpl::range<int, 0, 1000>>() == "list");
}

void test_list_of_names() {
  using empty_list = tmpl::list<>;
  using one_element_list = tmpl::list<Type1Containing2Digits3>;
  using three_element_list =
      tmpl::list<TestType, Type1Containing2Digits3, NonTemplateWithName>;

  CHECK(pretty_type::list_of_names<empty_list>() == "");
  CHECK(pretty_type::list_of_names<one_element_list>() ==
        "Type1Containing2Digits3");
  CHECK(pretty_type::list_of_names<three_element_list>() ==
        "TestType, Type1Containing2Digits3, UniqueNonTemplateWithName");
}

SPECTRE_TEST_CASE("Unit.Utilities.PrettyType.name_and_short_name",
                  "[Utilities][Unit]") {
  test_name_func<ShortName>();
  // For all these types without a name() member, the results should be
  // identical to that of pretty_type::short_name()
  test_name_func<Name>();
  test_list_of_names();

  CHECK(NonTemplateWithName::name() == "UniqueNonTemplateWithName");
  CHECK(TemplateWithName<int>::name() == "UniqueTemplateWithName");
}
}  // namespace
