// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <string>

#include "DataStructures/DataBox/ConstructibleFromTags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Empty {};
static_assert(not db::constructible_from_tags<Empty>);

struct IntTag {
  using type = int;
};

struct StringTag {
  using type = std::string;
};

struct Constructible {
  using creation_tags = tmpl::list<IntTag, StringTag, StringTag>;
  Constructible(int, std::string, const std::string&);
};
static_assert(db::constructible_from_tags<Constructible>);

struct NoTags {
  using creation_tags = tmpl::list<>;
};
static_assert(db::constructible_from_tags<NoTags>);

struct WrongArgs {
  using creation_tags = tmpl::list<StringTag>;
  explicit WrongArgs(int);
};
static_assert(not db::constructible_from_tags<WrongArgs>);

template <typename T>
struct TemplatedConstructible {
  using creation_tags = tmpl::list<IntTag, StringTag, StringTag>;
  TemplatedConstructible(int, std::string, const std::string&);
};
static_assert(db::constructible_from_tags<TemplatedConstructible<double>>);
}  // namespace
