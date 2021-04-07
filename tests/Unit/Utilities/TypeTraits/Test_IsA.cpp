// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <deque>
#include <forward_list>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/TypeTraits/IsA.hpp"

namespace {
class A;
class C;
class D;
} // namespace

// [is_a_example]
static_assert(tt::is_a<std::vector, std::vector<double>>::value,
              "Failed testing type trait is_a<vector>");
static_assert(not tt::is_a_t<std::vector, double>::value,
              "Failed testing type trait is_a<vector>");
static_assert(tt::is_a_v<std::vector, std::vector<D>>,
              "Failed testing type trait is_a<vector>");

static_assert(tt::is_a<std::deque, std::deque<double>>::value,
              "Failed testing type trait is_a<deque>");
static_assert(not tt::is_a<std::deque, double>::value,
              "Failed testing type trait is_a<deque>");
static_assert(tt::is_a<std::deque, std::deque<D>>::value,
              "Failed testing type trait is_a<deque>");

static_assert(tt::is_a<std::forward_list, std::forward_list<double>>::value,
              "Failed testing type trait is_a<forward_list>");
static_assert(not tt::is_a<std::forward_list, double>::value,
              "Failed testing type trait is_a<forward_list>");
static_assert(tt::is_a<std::forward_list, std::forward_list<D>>::value,
              "Failed testing type trait is_a<forward_list>");

static_assert(tt::is_a<std::list, std::list<double>>::value,
              "Failed testing type trait is_a<list>");
static_assert(not tt::is_a<std::list, double>::value,
              "Failed testing type trait is_a<list>");
static_assert(tt::is_a<std::list, std::list<D>>::value,
              "Failed testing type trait is_a<list>");

static_assert(tt::is_a<std::map, std::map<std::string, double>>::value,
              "Failed testing type trait is_a<map>");
static_assert(not tt::is_a<std::map, double>::value,
              "Failed testing type trait is_a<map>");
static_assert(tt::is_a<std::map, std::map<std::string, D>>::value,
              "Failed testing type trait is_a<map>");

static_assert(tt::is_a<std::unordered_map,
                       std::unordered_map<std::string, double>>::value,
              "Failed testing type trait is_a<unordered_map>");
static_assert(not tt::is_a<std::unordered_map, double>::value,
              "Failed testing type trait is_a<unordered_map>");
static_assert(
    tt::is_a<std::unordered_map, std::unordered_map<std::string, D>>::value,
    "Failed testing type trait is_a<unordered_map>");

static_assert(tt::is_a<std::set, std::set<double>>::value,
              "Failed testing type trait is_a<set>");
static_assert(not tt::is_a<std::set, double>::value,
              "Failed testing type trait is_a<set>");
static_assert(tt::is_a<std::set, std::set<D>>::value,
              "Failed testing type trait is_a<set>");

static_assert(tt::is_a<std::unordered_set, std::unordered_set<double>>::value,
              "Failed testing type trait is_a<unordered_set>");
static_assert(not tt::is_a<std::unordered_set, double>::value,
              "Failed testing type trait is_a<unordered_set>");
static_assert(tt::is_a<std::unordered_set, std::unordered_set<D>>::value,
              "Failed testing type trait is_a<unordered_set>");

static_assert(tt::is_a<std::multiset, std::multiset<double>>::value,
              "Failed testing type trait is_a<multiset>");
static_assert(not tt::is_a<std::multiset, double>::value,
              "Failed testing type trait is_a<multiset>");
static_assert(tt::is_a<std::multiset, std::multiset<D>>::value,
              "Failed testing type trait is_a<multiset>");

static_assert(
    tt::is_a<std::unordered_multiset, std::unordered_multiset<double>>::value,
    "Failed testing type trait is_a<unordered_multiset>");
static_assert(not tt::is_a<std::unordered_multiset, double>::value,
              "Failed testing type trait is_a<unordered_multiset>");
static_assert(
    tt::is_a<std::unordered_multiset, std::unordered_multiset<D>>::value,
    "Failed testing type trait is_a<unordered_multiset>");

static_assert(
    tt::is_a<std::multimap, std::multimap<std::string, double>>::value,
    "Failed testing type trait is_a<multimap>");
static_assert(not tt::is_a<std::multimap, double>::value,
              "Failed testing type trait is_a<multimap>");
static_assert(tt::is_a<std::multimap, std::multimap<std::string, D>>::value,
              "Failed testing type trait is_a<multimap>");

static_assert(tt::is_a<std::unordered_multimap,
                       std::unordered_multimap<std::string, double>>::value,
              "Failed testing type trait is_a<unordered_multimap>");
static_assert(not tt::is_a<std::unordered_multimap, double>::value,
              "Failed testing type trait is_a<unordered_multimap>");
static_assert(tt::is_a<std::unordered_multimap,
                       std::unordered_multimap<std::string, D>>::value,
              "Failed testing type trait is_a<unordered_multimap>");

static_assert(tt::is_a<std::priority_queue, std::priority_queue<double>>::value,
              "Failed testing type trait is_a<priority_queue>");
static_assert(not tt::is_a<std::priority_queue, double>::value,
              "Failed testing type trait is_a<priority_queue>");
static_assert(tt::is_a<std::priority_queue, std::priority_queue<D>>::value,
              "Failed testing type trait is_a<priority_queue>");

static_assert(tt::is_a<std::queue, std::queue<double>>::value,
              "Failed testing type trait is_a<queue>");
static_assert(not tt::is_a<std::queue, double>::value,
              "Failed testing type trait is_a<queue>");
static_assert(tt::is_a<std::queue, std::queue<D>>::value,
              "Failed testing type trait is_a<queue>");

static_assert(tt::is_a<std::stack, std::stack<double>>::value,
              "Failed testing type trait is_a<stack>");
static_assert(not tt::is_a<std::stack, double>::value,
              "Failed testing type trait is_a<stack>");
static_assert(tt::is_a<std::stack, std::stack<D>>::value,
              "Failed testing type trait is_a<stack>");

static_assert(tt::is_a<std::unique_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<unique_ptr>");
static_assert(tt::is_a<std::unique_ptr, std::unique_ptr<C>>::value,
              "Failed testing type trait is_a<unique_ptr>");
static_assert(not tt::is_a<std::unique_ptr, std::shared_ptr<double>>::value,
              "Failed testing type trait is_a<unique_ptr>");
static_assert(not tt::is_a<std::unique_ptr, C>::value,
              "Failed testing type trait is_a<unique_ptr>");

static_assert(tt::is_a<std::shared_ptr, std::shared_ptr<double>>::value,
              "Failed testing type trait is_a<shared_ptr>");
static_assert(tt::is_a<std::shared_ptr, std::shared_ptr<C>>::value,
              "Failed testing type trait is_a<shared_ptr>");
static_assert(not tt::is_a<std::shared_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<shared_ptr>");
static_assert(not tt::is_a<std::shared_future, C>::value,
              "Failed testing type trait is_a<shared_ptr>");

static_assert(tt::is_a<std::weak_ptr, std::weak_ptr<double>>::value,
              "Failed testing type trait is_a<weak_ptr>");
static_assert(tt::is_a<std::weak_ptr, std::weak_ptr<C>>::value,
              "Failed testing type trait is_a<weak_ptr>");
static_assert(not tt::is_a<std::weak_ptr, std::unique_ptr<double>>::value,
              "Failed testing type trait is_a<weak_ptr>");
static_assert(not tt::is_a<std::weak_ptr, C>::value,
              "Failed testing type trait is_a<weak_ptr>");

static_assert(tt::is_a<std::tuple, std::tuple<int, double, A>>::value,
              "Failed testing type trait is_a");
static_assert(tt::is_a<std::vector, std::vector<A>>::value,
              "Failed testing type trait is_a");

static_assert(tt::is_a<std::future, std::future<double>>::value,
              "Failed testing type trait is_a<future>");
static_assert(tt::is_a<std::future, std::future<std::vector<double>>>::value,
              "Failed testing type trait is_a<future>");
static_assert(not tt::is_a<std::future, std::shared_future<double>>::value,
              "Failed testing type trait is_a<future>");

static_assert(tt::is_a<std::shared_future, std::shared_future<double>>::value,
              "Failed testing type trait is_a<shared_future>");
static_assert(tt::is_a<std::shared_future,
                       std::shared_future<std::vector<double>>>::value,
              "Failed testing type trait is_a<shared_future>");
static_assert(not tt::is_a<std::shared_future, std::future<double>>::value,
              "Failed testing type trait is_a<shared_future>");
// [is_a_example]
