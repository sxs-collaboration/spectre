// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/Serialization/Serializable.hpp"

namespace {
class A {};
static_assert(not serializable<A>);

class B {
 public:
  void pup(PUP::er& p);
};
static_assert(serializable<B>);

class C {
 public:
  explicit C(int);
  void pup(PUP::er& p);
};
static_assert(not serializable<C>);

static_assert(serializable<int>);

enum class Enum { X };
static_assert(serializable<Enum>);
}  // namespace
