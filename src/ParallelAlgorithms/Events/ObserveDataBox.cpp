// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveDataBox.hpp"

#include <fstream>
#include <pup.h>

#include "DataStructures/DataBox/Access.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Events {
ObserveDataBox::ObserveDataBox(
    std::optional<std::string> file_name_for_tag_names)
    : file_name_for_tag_names_(std::move(file_name_for_tag_names)) {}

void ObserveDataBox::pup(PUP::er& p) {
  Event::pup(p);
  p | file_name_for_tag_names_;
}

template <size_t VolumeDim>
void ObserveDataBox::impl(const db::Access& box_access,
                          const ElementId<VolumeDim>& array_index,
                          const ObservationValue& /*observation_value*/) const {
  if (is_zeroth_element(array_index)) {
    if (file_name_for_tag_names_.has_value()) {
      std::ofstream of{file_name_for_tag_names_.value()};
      of << box_access.print_tags() << "\n";
      of.close();
    }
  }
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID ObserveDataBox::my_PUP_ID = 0;  // NOLINT
#endif                                            // SPECTRE_USE_CHARM

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                                   \
  template void ObserveDataBox::impl<DIM(data)>(                               \
      const db::Access&, const ElementId<DIM(data)>&, const ObservationValue&) \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace Events
