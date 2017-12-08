// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/BoostHelpers.hpp"
#include "Utilities/TMPL.hpp"

static_assert(
    cpp17::is_same_v<boost::variant<double, int, char>,
                     make_boost_variant_over<typelist<double, int, char>>>,
    "Failed testing make_variant_over");
