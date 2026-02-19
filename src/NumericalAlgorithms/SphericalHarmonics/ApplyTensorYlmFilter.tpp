// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm::TensorYlm::filter_detail {

template <typename VarsList>
void nodal_to_modal_ylm(const gsl::not_null<Variables<VarsList>*> modal,
                        const Variables<VarsList>& nodal,
                        const ::ylm::Spherepack& ylm,
                        const size_t radial_extents) {
  tmpl::for_each<VarsList>([&nodal, &modal, radial_extents,
                            &ylm]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
    constexpr size_t num_independent_components = Tag::type::structure::size();
    for (size_t storage_index = 0; storage_index < num_independent_components;
         ++storage_index) {
      const double* src = get<Tag>(nodal)[storage_index].data();
      double* dest = get<Tag>(*modal)[storage_index].data();
      ylm.phys_to_spec_all_offsets(make_not_null(dest), make_not_null(src),
                                   radial_extents);
    }
  });
}

template <typename VarsList>
void modal_to_nodal_ylm(const gsl::not_null<Variables<VarsList>*> nodal,
                        const Variables<VarsList>& modal,
                        const ::ylm::Spherepack& ylm,
                        const size_t radial_extents) {
  tmpl::for_each<VarsList>([&nodal, &modal, radial_extents,
                            &ylm]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
    constexpr size_t num_independent_components = Tag::type::structure::size();
    for (size_t storage_index = 0; storage_index < num_independent_components;
         ++storage_index) {
      const double* src = get<Tag>(modal)[storage_index].data();
      double* dest = get<Tag>(*nodal)[storage_index].data();
      ylm.spec_to_phys_all_offsets(make_not_null(dest), make_not_null(src),
                                   radial_extents);
    }
  });
}
}  // namespace ylm::TensorYlm::filter_detail
