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
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg::subcell {
SubcellOptions::SubcellOptions(
    double initial_data_rdmp_delta0, double initial_data_rdmp_epsilon,
    double rdmp_delta0, double rdmp_epsilon,
    double initial_data_persson_exponent, double persson_exponent,
    bool always_use_subcells, fd::ReconstructionMethod recons_method,
    bool use_halo,
    std::optional<std::vector<std::string>> only_dg_block_and_group_names,
    ::fd::DerivativeOrder finite_difference_derivative_order)
    : initial_data_rdmp_delta0_(initial_data_rdmp_delta0),
      initial_data_rdmp_epsilon_(initial_data_rdmp_epsilon),
      rdmp_delta0_(rdmp_delta0),
      rdmp_epsilon_(rdmp_epsilon),
      initial_data_persson_exponent_(initial_data_persson_exponent),
      persson_exponent_(persson_exponent),
      always_use_subcells_(always_use_subcells),
      reconstruction_method_(recons_method),
      use_halo_(use_halo),
      only_dg_block_and_group_names_(std::move(only_dg_block_and_group_names)),
      finite_difference_derivative_order_(finite_difference_derivative_order) {
  if (not only_dg_block_and_group_names_.has_value()) {
    only_dg_block_ids_ = std::vector<size_t>{};
  }
}

template <size_t Dim>
SubcellOptions::SubcellOptions(
    const SubcellOptions& subcell_options_with_block_names,
    const DomainCreator<Dim>& domain_creator) {
  *this = subcell_options_with_block_names;

  const auto& only_dg_block_and_group_names =
      subcell_options_with_block_names.only_dg_block_and_group_names_;
  const auto block_names_vector = domain_creator.block_names();
  const auto group_names = domain_creator.block_groups();
  std::unordered_set<std::string> block_names{};
  // Add blocks from block groups into block_names. Use an unordered_set
  // to elide duplicates.
  for (const auto& block_or_group_name :
       only_dg_block_and_group_names.value_or(std::vector<std::string>{})) {
    if (const auto block_group_it = group_names.find(block_or_group_name);
        block_group_it != group_names.end()) {
      block_names.insert(block_group_it->second.begin(),
                         block_group_it->second.end());
    } else if (const auto block_name_it =
                   std::find(block_names_vector.begin(),
                             block_names_vector.end(), block_or_group_name);
               block_name_it != block_names_vector.end()) {
      block_names.insert(*block_name_it);
    } else {
      using ::operator<<;
      ERROR_NO_TRACE(
          "The block or group '"
          << block_or_group_name
          << "' is not one of the block names or groups of the domain. "
             "The known block groups are "
          << keys_of(group_names) << " and the known block names are "
          << block_names_vector);
    }
  }
  only_dg_block_ids_ = std::vector<size_t>{};
  only_dg_block_ids_.value().reserve(block_names.size());
  // Get the block ID of each block name
  for (const auto& block_name : block_names) {
    only_dg_block_ids_.value().push_back(static_cast<size_t>(
        std::distance(block_names_vector.begin(),
                      std::find(block_names_vector.begin(),
                                block_names_vector.end(), block_name))));
  }
  // Sort the block IDs just so they're easier to deal with.
  alg::sort(only_dg_block_ids_.value());
}

void SubcellOptions::pup(PUP::er& p) {
  p | initial_data_rdmp_delta0_;
  p | initial_data_rdmp_epsilon_;
  p | rdmp_delta0_;
  p | rdmp_epsilon_;
  p | initial_data_persson_exponent_;
  p | persson_exponent_;
  p | always_use_subcells_;
  p | reconstruction_method_;
  p | use_halo_;
  p | only_dg_block_and_group_names_;
  p | only_dg_block_ids_;
  p | finite_difference_derivative_order_;
}

bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs) {
  return lhs.initial_data_rdmp_delta0() == rhs.initial_data_rdmp_delta0() and
         lhs.initial_data_rdmp_epsilon() == rhs.initial_data_rdmp_epsilon() and
         lhs.rdmp_delta0() == rhs.rdmp_delta0() and
         lhs.rdmp_epsilon() == rhs.rdmp_epsilon() and
         lhs.initial_data_persson_exponent() ==
             rhs.initial_data_persson_exponent() and
         lhs.persson_exponent() == rhs.persson_exponent() and
         lhs.always_use_subcells() == rhs.always_use_subcells() and
         lhs.reconstruction_method() == rhs.reconstruction_method() and
         lhs.use_halo() == rhs.use_halo() and
         lhs.only_dg_block_and_group_names_ ==
             rhs.only_dg_block_and_group_names_ and
         lhs.only_dg_block_ids_ == rhs.only_dg_block_ids_ and
         lhs.finite_difference_derivative_order_ ==
             rhs.finite_difference_derivative_order_;
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
