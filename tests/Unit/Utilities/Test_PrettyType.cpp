// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/PrettyType.hpp"

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

SPECTRE_TEST_CASE("Unit.Utilities.PrettyType.short_name", "[Utilities][Unit]") {
  CHECK("Simple" == pretty_type::extract_short_name("Simple"));
  CHECK("Qualified" == pretty_type::extract_short_name("Namespace::Qualified"));
  CHECK("Templated" == pretty_type::extract_short_name("Templated<int>"));
  CHECK("Nested" == pretty_type::extract_short_name("Nested<Templated<int>>"));
  CHECK("Nested" ==
        pretty_type::extract_short_name("Nested<Namespace::Templated<int>>"));
  CHECK("Inner" == pretty_type::extract_short_name("Outer<int>::Inner"));
}
