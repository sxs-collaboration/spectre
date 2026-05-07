// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg::Initialization {
template <size_t Dim, typename TagList>
void SpectralFilters<Dim, TagList>::apply(
    const gsl::not_null<std::unique_ptr<Filters::Filter<Dim, TagList>>*>
        spectral_filter,
    const std::vector<std::unique_ptr<Filters::Filter<Dim, TagList>>>&
        spectral_filters,
    const Element<Dim>& element, const Mesh<Dim>& mesh) {
  *spectral_filter = nullptr;
  for (const std::unique_ptr<Filters::Filter<Dim, TagList>>& filter :
       spectral_filters) {
    if (const auto& blocks_to_filter = filter->blocks_to_filter();
        filter->supports_mesh(mesh) and
        ((not blocks_to_filter.has_value()) or
         alg::found(blocks_to_filter.value(), element.id().block_id()))) {
      if (*spectral_filter != nullptr) {
        ERROR("Cannot specify more than one filter for element "
              << element.id() << ".");
      }
      *spectral_filter = filter->get_clone();
    }
  }
  if (*spectral_filter == nullptr) {
    ERROR("No filter found for element "
          << element.id() << " with basis " << mesh.basis()
          << ". You can specify the None filter to disable filtering in an "
             "element.");
  }
}
}  // namespace evolution::dg::Initialization
