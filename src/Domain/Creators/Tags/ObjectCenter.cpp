// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Tags/ObjectCenter.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain::Tags {
template <ObjectLabel Label>
tnsr::I<double, 3, Frame::Grid> ExcisionCenter<Label>::create_from_options(
    const std::unique_ptr<::DomainCreator<3>>& domain_creator) {
  const auto domain = domain_creator->create_domain();
  const std::string name = "ExcisionSphere"s + get_output(Label);
  if (domain.excision_spheres().count(name) != 1) {
    ERROR(name << " is not in the domains excision spheres but is needed to "
                  "generate the ExcisionCenter<"
               << Label << ">.");
  }

  return domain.excision_spheres().at(name).center();
}

#define OBJECT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                  \
  template struct ObjectCenter<OBJECT(data)>; \
  template struct ExcisionCenter<OBJECT(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (ObjectLabel::A, ObjectLabel::B, ObjectLabel::None))

#undef INSTANTIATE
#undef OBJECT
}  // namespace domain::Tags
