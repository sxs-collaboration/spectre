// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// [conforming_type_example]
struct ValidNumericInitialData
    : tt::ConformsTo<evolution::protocols::NumericInitialData> {
  using import_fields = tmpl::list<FieldTag>;
};
// [conforming_type_example]
struct InvalidNumericInitialData
    : tt::ConformsTo<evolution::protocols::NumericInitialData> {};

static_assert(
    evolution::protocols::NumericInitialData<ValidNumericInitialData>::value,
    "Failed testing protocol");
static_assert(not evolution::protocols::NumericInitialData<
                  InvalidNumericInitialData>::value,
              "Failed testing protocol");

}  // namespace
