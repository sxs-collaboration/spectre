// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Initialize.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Var>
struct Sector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var>;

  struct initial_guess {
    using return_tags = tmpl::list<Var>;
    using argument_tags = tmpl::list<>;
    static imex::GuessResult apply(
        gsl::not_null<Scalar<DataVector>*> var,
        const Variables<tmpl::list<Var>>& inhomogeneous_terms,
        double implicit_weight);
  };

  struct SolveAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var>>;
      using argument_tags = tmpl::list<Var>;
      static void apply(gsl::not_null<Scalar<DataVector>*> source);
    };

    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;

    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };
  using solve_attempts = tmpl::list<SolveAttempt>;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using implicit_sectors = tmpl::list<Sector<Var1>, Sector<Var2>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Initialize", "[Unit][Evolution]") {
  using initialize_imex = imex::Initialize<System>;
  using explicit_history_tag =
      Tags::HistoryEvolvedVariables<System::variables_tag>;
  auto box =
      db::create<db::AddSimpleTags<System::variables_tag, explicit_history_tag,
                                   initialize_imex::simple_tags>>();
  db::mutate<System::variables_tag, explicit_history_tag>(
      [](const gsl::not_null<System::variables_tag::type*> variables,
         const gsl::not_null<explicit_history_tag::type*> explicit_history) {
        variables->initialize(5);
        explicit_history->integration_order(3);
      },
      make_not_null(&box));

  db::mutate_apply<initialize_imex>(make_not_null(&box));

  CHECK(db::get<imex::Tags::ImplicitHistory<Sector<Var1>>>(box)
            .integration_order() == 3);
  CHECK(db::get<imex::Tags::ImplicitHistory<Sector<Var2>>>(box)
            .integration_order() == 3);
  CHECK(db::get<imex::Tags::SolveFailures<Sector<Var1>>>(box) ==
        Scalar<DataVector>(DataVector(5, 0.0)));
  CHECK(db::get<imex::Tags::SolveFailures<Sector<Var2>>>(box) ==
        Scalar<DataVector>(DataVector(5, 0.0)));
}
