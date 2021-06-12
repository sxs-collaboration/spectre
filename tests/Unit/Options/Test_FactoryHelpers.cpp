// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "Options/FactoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Use a type for the label instead of an integer to catch any missing
// tmpl::pins or such.
template <typename Label>
struct Base;
template <typename Label>
struct Derived;
template <int>
struct Label;

using original_classes =
    tmpl::map<tmpl::pair<Base<Label<0>>,
                         tmpl::list<Derived<Label<0>>, Derived<Label<1>>>>,
              tmpl::pair<Base<Label<1>>, tmpl::list<Derived<Label<2>>>>>;

static_assert(std::is_same_v<original_classes,
                             Options::add_factory_classes<original_classes>>);

// [add_factory_classes]
using new_classes = Options::add_factory_classes<
    original_classes,
    tmpl::pair<Base<Label<0>>,
               tmpl::list<Derived<Label<3>>, Derived<Label<4>>>>,
    tmpl::pair<Base<Label<2>>, tmpl::list<Derived<Label<5>>>>>;
// [add_factory_classes]

// tmpl::map doesn't guarantee an order for the keys, so we have to
// check each entry separately.
static_assert(tmpl::size<new_classes>::value == 3);
static_assert(std::is_same_v<tmpl::at<new_classes, Base<Label<0>>>,
                             tmpl::list<Derived<Label<0>>, Derived<Label<1>>,
                                        Derived<Label<3>>, Derived<Label<4>>>>);
static_assert(std::is_same_v<tmpl::at<new_classes, Base<Label<1>>>,
                             tmpl::list<Derived<Label<2>>>>);
static_assert(std::is_same_v<tmpl::at<new_classes, Base<Label<2>>>,
                             tmpl::list<Derived<Label<5>>>>);
}  // namespace
