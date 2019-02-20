// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"       // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare SpinWeighted
// IWYU pragma: no_forward_declare Tensor

/// \cond
class ComplexDataVector;
class ComplexModalVector;
/// \endcond

namespace Spectral {
namespace Swsh {
namespace Tags {
namespace {

struct UnweightedTestTag : db::SimpleTag {
  static std::string name() noexcept { return "UnweightedTestTag"; }
  using type = Scalar<ComplexDataVector>;
};

struct SpinMinus1TestTag : db::SimpleTag {
  static std::string name() noexcept { return "SpinMinus1TestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

struct AnotherSpinMinus1TestTag : db::SimpleTag {
  static std::string name() noexcept { return "AnotherSpinMinus1TestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

struct Spin2TestTag : db::SimpleTag {
  static std::string name() noexcept { return "Spin2TestTag"; }
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
};

static_assert(Derivative<Spin2TestTag, Ethbar>::spin == 1,
              "failed testing DerivativeTag with DerivativeType Ethbar");

static_assert(
    cpp17::is_same_v<Derivative<SpinMinus1TestTag, Ethbar>::derived_from,
                     SpinMinus1TestTag>,
    "failed testing DerivativeTag with DerivativeType Ethbar");

static_assert(cpp17::is_same_v<SwshTransform<Spin2TestTag>::type,
                               Scalar<SpinWeighted<ComplexModalVector, 2>>>,
              "failed testing SwshTransform");

static_assert(
    cpp17::is_same_v<
        SwshTransform<Derivative<Spin2TestTag, EthEthbar>>::transform_of,
        Derivative<Spin2TestTag, EthEthbar>>,
    "failed testing SwshTransform");

/// [get_tags_with_spin]
using TestVarTagList = tmpl::list<SpinMinus1TestTag, SpinMinus1TestTag,
                                  AnotherSpinMinus1TestTag, Spin2TestTag>;

static_assert(
    cpp17::is_same_v<get_tags_with_spin<-1, TestVarTagList>,
                     tmpl::list<SpinMinus1TestTag, AnotherSpinMinus1TestTag>>,
    "failed testing get_tags_with_spin");

using TestDerivativeTagList =
    tmpl::list<Derivative<SpinMinus1TestTag, Eth>,
               Derivative<SpinMinus1TestTag, EthEthbar>,
               Derivative<AnotherSpinMinus1TestTag, EthEth>,
               Derivative<Spin2TestTag, Ethbar>>;

static_assert(
    cpp17::is_same_v<get_tags_with_spin<1, TestDerivativeTagList>,
                     tmpl::list<Derivative<AnotherSpinMinus1TestTag, EthEth>,
                                Derivative<Spin2TestTag, Ethbar>>>,
    "failed testing get_tags_with_spin");
/// [get_tags_with_spin]

/// [get_prefix_tags_that_wrap_tags_with_spin]
using WrappedTagList = tmpl::list<Derivative<SpinMinus1TestTag, Eth>,
                                  Derivative<SpinMinus1TestTag, EthEthbar>,
                                  Derivative<AnotherSpinMinus1TestTag, EthEth>,
                                  Derivative<Spin2TestTag, Ethbar>>;
static_assert(cpp17::is_same_v<
                  get_prefix_tags_that_wrap_tags_with_spin<-1, WrappedTagList>,
                  tmpl::list<Derivative<SpinMinus1TestTag, Eth>,
                             Derivative<SpinMinus1TestTag, EthEthbar>,
                             Derivative<AnotherSpinMinus1TestTag, EthEth>>>,
              "failed testing get_wrapped_tags_with_spin_from_prefix_tag_list");
/// [get_prefix_tags_that_wrap_tags_with_spin]

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Tags",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(Derivative<SpinMinus1TestTag, Eth>::name() == "Eth(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, EthEth>::name() ==
        "EthEth(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, EthEthbar>::name() ==
        "EthEthbar(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, Ethbar>::name() ==
        "Ethbar(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, EthbarEth>::name() ==
        "EthbarEth(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, EthbarEthbar>::name() ==
        "EthbarEthbar(SpinMinus1TestTag)");
  CHECK(Derivative<SpinMinus1TestTag, NoDerivative>::name() ==
        "NoDerivative(SpinMinus1TestTag)");
  CHECK(SwshTransform<Spin2TestTag>::name() == "SwshTransform(Spin2TestTag)");
}
}  // namespace
}  // namespace Tags
}  // namespace Swsh
}  // namespace Spectral
