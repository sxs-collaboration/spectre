// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SubcellOptions.hpp"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg::subcell {
SubcellOptions::SubcellOptions(
    double persson_exponent, size_t persson_num_highest_modes,
    double rdmp_delta0, double rdmp_epsilon, bool always_use_subcells,
    fd::ReconstructionMethod recons_method, bool use_halo,
    std::optional<std::vector<std::string>> only_dg_block_and_group_names,
    ::fd::DerivativeOrder finite_difference_derivative_order,
    const size_t number_of_steps_between_tci_calls,
    const size_t min_tci_calls_after_rollback,
    const size_t min_clear_tci_before_dg)
    : persson_exponent_(persson_exponent),
      persson_num_highest_modes_(persson_num_highest_modes),
      rdmp_delta0_(rdmp_delta0),
      rdmp_epsilon_(rdmp_epsilon),
      always_use_subcells_(always_use_subcells),
      reconstruction_method_(recons_method),
      use_halo_(use_halo),
      only_dg_block_and_group_names_(std::move(only_dg_block_and_group_names)),
      finite_difference_derivative_order_(finite_difference_derivative_order),
      number_of_steps_between_tci_calls_(number_of_steps_between_tci_calls),
      min_tci_calls_after_rollback_(min_tci_calls_after_rollback),
      min_clear_tci_before_dg_(min_clear_tci_before_dg) {
  if (not only_dg_block_and_group_names_.has_value()) {
    only_dg_block_ids_ = std::vector<size_t>{};
  }
  ASSERT(number_of_steps_between_tci_calls_ > 0,
         "number_of_steps_between_tci_calls_ must be greater than zero.");
  ASSERT(min_tci_calls_after_rollback_ > 0,
         "min_tci_calls_after_rollback_ must be greater than zero.");
  ASSERT(min_clear_tci_before_dg_ > 0,
         "min_clear_tci_before_dg_ must be greater than zero.");
}

template <size_t Dim>
SubcellOptions::SubcellOptions(
    const SubcellOptions& subcell_options_with_block_names,
    const DomainCreator<Dim>& domain_creator) {
  *this = subcell_options_with_block_names;

  const auto& only_dg_block_and_group_names =
      subcell_options_with_block_names.only_dg_block_and_group_names_;
  const auto block_names = domain_creator.block_names();
  const auto only_dg_block_names = domain::expand_block_groups_to_block_names(
      only_dg_block_and_group_names.value_or(std::vector<std::string>{}),
      block_names, domain_creator.block_groups());
  only_dg_block_ids_ = std::vector<size_t>{};
  only_dg_block_ids_.value().reserve(only_dg_block_names.size());
  // Get the block ID of each block name
  for (const auto& block_name : only_dg_block_names) {
    only_dg_block_ids_.value().push_back(static_cast<size_t>(std::distance(
        block_names.begin(),
        std::find(block_names.begin(), block_names.end(), block_name))));
  }
  // Sort the block IDs just so they're easier to deal with.
  alg::sort(only_dg_block_ids_.value());
}

void SubcellOptions::pup(PUP::er& p) {
  p | persson_exponent_;
  p | persson_num_highest_modes_;
  p | rdmp_delta0_;
  p | rdmp_epsilon_;
  p | always_use_subcells_;
  p | reconstruction_method_;
  p | use_halo_;
  p | only_dg_block_and_group_names_;
  p | only_dg_block_ids_;
  p | finite_difference_derivative_order_;
  p | number_of_steps_between_tci_calls_;
  p | min_tci_calls_after_rollback_;
  p | min_clear_tci_before_dg_;
}

bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs) {
  return lhs.persson_exponent() == rhs.persson_exponent() and
         lhs.persson_num_highest_modes() == rhs.persson_num_highest_modes() and
         lhs.rdmp_delta0() == rhs.rdmp_delta0() and
         lhs.rdmp_epsilon() == rhs.rdmp_epsilon() and
         lhs.always_use_subcells() == rhs.always_use_subcells() and
         lhs.reconstruction_method() == rhs.reconstruction_method() and
         lhs.use_halo() == rhs.use_halo() and
         lhs.only_dg_block_and_group_names_ ==
             rhs.only_dg_block_and_group_names_ and
         lhs.only_dg_block_ids_ == rhs.only_dg_block_ids_ and
         lhs.finite_difference_derivative_order_ ==
             rhs.finite_difference_derivative_order_ and
         lhs.number_of_steps_between_tci_calls_ ==
             rhs.number_of_steps_between_tci_calls_ and
         lhs.min_tci_calls_after_rollback_ ==
             rhs.min_tci_calls_after_rollback_ and
         lhs.min_clear_tci_before_dg_ == rhs.min_clear_tci_before_dg_;
}

bool operator!=(const SubcellOptions& lhs, const SubcellOptions& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template SubcellOptions::SubcellOptions(                    \
      const SubcellOptions& subcell_options_with_block_names, \
      const DomainCreator<DIM(data)>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
