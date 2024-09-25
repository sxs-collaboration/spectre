// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImexSystem.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Protocols/ImplicitSource.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

// Separated from Test_DoImplicitStep.cpp to test explicit
// instantiation of SolveSimplcitSector (in
// DoImplicitStepInstantiate.cpp).
namespace do_implicit_step_helpers {
template <typename Var>
// [simple_sector]
struct Sector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var>;

  struct initial_guess {
    using return_tags = tmpl::list<Var>;
    using argument_tags = tmpl::list<>;
    static std::vector<imex::GuessResult> apply(
        const gsl::not_null<Scalar<DataVector>*> var,
        const Variables<tmpl::list<Var>>& inhomogeneous_terms,
        const double implicit_weight) {
      get(*var) = get(get<Var>(inhomogeneous_terms)) / (1.0 + implicit_weight);
      return {get(*var).size(), imex::GuessResult::ExactSolution};
    }
  };

  struct SolveAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var>>;
      using argument_tags = tmpl::list<Var>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> source,
                        const Scalar<DataVector>& var) {
        get(*source) = -get(var);
      }
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
// [simple_sector]

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// [ImexSystem]
struct System : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;

  // Explicit evolution stuff here...

  using implicit_sectors = tmpl::list<Sector<Var1>, Sector<Var2>>;
};
// [ImexSystem]

struct NonautonomousSector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<Var1>;

  struct initial_guess {
    using return_tags = tmpl::list<Var1>;
    using argument_tags = tmpl::list<::Tags::Time>;
    static std::vector<imex::GuessResult> apply(
        const gsl::not_null<Scalar<DataVector>*> var, const double time,
        const Variables<tmpl::list<Var1>>& inhomogeneous_terms,
        const double implicit_weight) {
      get(*var) = get(get<Var1>(inhomogeneous_terms)) + implicit_weight * time;
      return {get(*var).size(), imex::GuessResult::ExactSolution};
    }
  };

  struct SolveAttempt {
    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<Tags::Source<Var1>>;
      using argument_tags = tmpl::list<::Tags::Time>;
      static void apply(const gsl::not_null<Scalar<DataVector>*> source,
                        const double time) {
        get(*source) = time;
      }
    };

    using jacobian = imex::NoJacobianBecauseSolutionIsAnalytic;

    using tags_from_evolution = tmpl::list<::Tags::Time>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;
  };

  using solve_attempts = tmpl::list<SolveAttempt>;
};

struct NonautonomousSystem : tt::ConformsTo<imex::protocols::ImexSystem> {
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
  using implicit_sectors = tmpl::list<NonautonomousSector>;
};
}  // namespace do_implicit_step_helpers
