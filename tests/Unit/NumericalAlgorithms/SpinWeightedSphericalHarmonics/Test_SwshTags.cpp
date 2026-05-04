// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

class ComplexDataVector;
class ComplexModalVector;

namespace Spectral::Swsh::Tags {
namespace {

struct UnweightedTestTag : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

struct SpinMinus1TestTag : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

struct AnotherSpinMinus1TestTag : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, -1>>;
};

struct Spin2TestTag : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 2>>;
};

static_assert(Derivative<Spin2TestTag, Ethbar>::spin == 1,
              "failed testing DerivativeTag with DerivativeType Ethbar");

static_assert(
    std::is_same_v<Derivative<SpinMinus1TestTag, Ethbar>::derivative_of,
                   SpinMinus1TestTag>,
    "failed testing DerivativeTag with DerivativeType Ethbar");

static_assert(std::is_same_v<SwshTransform<Spin2TestTag>::type,
                             Scalar<SpinWeighted<ComplexModalVector, 2>>>,
              "failed testing SwshTransform");

static_assert(
    std::is_same_v<
        SwshTransform<Derivative<Spin2TestTag, EthEthbar>>::transform_of,
        Derivative<Spin2TestTag, EthEthbar>>,
    "failed testing SwshTransform");

namespace test_spins_in_tag_list {
// [spins_in_tag_list]
using TestVarTagList = tmpl::list<SpinMinus1TestTag, SpinMinus1TestTag,
                                  Spin2TestTag, AnotherSpinMinus1TestTag>;

static_assert(std::is_same_v<spins_in_tag_list<TestVarTagList>,
                             tmpl::list<std::integral_constant<int, -1>,
                                        std::integral_constant<int, 2>>>);

using TestDerivativeTagList =
    tmpl::list<Derivative<SpinMinus1TestTag, Eth>,
               Derivative<SpinMinus1TestTag, EthEthbar>,
               Derivative<AnotherSpinMinus1TestTag, EthEth>,
               Derivative<Spin2TestTag, Ethbar>>;

static_assert(std::is_same_v<spins_in_tag_list<TestDerivativeTagList>,
                             tmpl::list<std::integral_constant<int, -1>,
                                        std::integral_constant<int, 0>,
                                        std::integral_constant<int, 1>>>);
// [spins_in_tag_list]
}  // namespace test_spins_in_tag_list

namespace test_partition_tags_by_spin {
// [partition_tags_by_spin]
using TestVarTagList = tmpl::list<SpinMinus1TestTag, SpinMinus1TestTag,
                                  Spin2TestTag, AnotherSpinMinus1TestTag>;

static_assert(
    std::is_same_v<partition_tags_by_spin<TestVarTagList>,
                   tmpl::list<tmpl::list<SpinMinus1TestTag, SpinMinus1TestTag,
                                         AnotherSpinMinus1TestTag>,
                              tmpl::list<Spin2TestTag>>>);

using TestDerivativeTagList =
    tmpl::list<Derivative<SpinMinus1TestTag, Eth>,
               Derivative<SpinMinus1TestTag, EthEthbar>,
               Derivative<AnotherSpinMinus1TestTag, EthEth>,
               Derivative<Spin2TestTag, Ethbar>>;

static_assert(
    std::is_same_v<
        partition_tags_by_spin<TestDerivativeTagList>,
        tmpl::list<tmpl::list<Derivative<SpinMinus1TestTag, EthEthbar>>,
                   tmpl::list<Derivative<SpinMinus1TestTag, Eth>>,
                   tmpl::list<Derivative<AnotherSpinMinus1TestTag, EthEth>,
                              Derivative<Spin2TestTag, Ethbar>>>>);
// [partition_tags_by_spin]
}  // namespace test_partition_tags_by_spin

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Tags",
                  "[Unit][NumericalAlgorithms]") {
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, Eth>>(
      "Eth(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, EthEth>>(
      "EthEth(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, EthEthbar>>(
      "EthEthbar(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, Ethbar>>(
      "Ethbar(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, EthbarEth>>(
      "EthbarEth(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, EthbarEthbar>>(
      "EthbarEthbar(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<Derivative<SpinMinus1TestTag, NoDerivative>>(
      "NoDerivative(SpinMinus1TestTag)");
  TestHelpers::db::test_prefix_tag<SwshTransform<Spin2TestTag>>(
      "SwshTransform(Spin2TestTag)");
  TestHelpers::db::test_prefix_tag<SwshInterpolator<Spin2TestTag>>(
      "SwshInterpolator(Spin2TestTag)");
  TestHelpers::db::test_simple_tag<LMax>("LMax");
  TestHelpers::db::test_simple_tag<NumberOfRadialPoints>(
      "NumberOfRadialPoints");
  CHECK(TestHelpers::test_option_tag<OptionTags::LMax>("8") == 8_st);
  CHECK(TestHelpers::test_option_tag<OptionTags::NumberOfRadialPoints>("3") ==
        3_st);
}
}  // namespace
}  // namespace Spectral::Swsh::Tags
