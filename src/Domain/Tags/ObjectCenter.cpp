// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Tags/ObjectCenter.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ObjectLabel.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain::Tags {
template <ObjectLabel Label>
tnsr::I<double, 3, Frame::Grid> ExcisionCenter<Label>::create_from_options(
    const std::unique_ptr<::DomainCreator<3>>& domain_creator) {
  const auto domain = domain_creator->create_domain();
  const std::string name = "Object"s + get_output(Label) + "ExcisionSphere"s;
  if (domain.excision_spheres().count(name) != 1) {
    ERROR(name << " is not in the domains excision spheres but is needed to "
                  "generate the ExcisionCenter<"
               << Label << ">.");
  }

  return domain.excision_spheres().at(name).center();
}

template struct ObjectCenter<ObjectLabel::A>;
template struct ObjectCenter<ObjectLabel::B>;
template struct ExcisionCenter<ObjectLabel::A>;
template struct ExcisionCenter<ObjectLabel::B>;
}  // namespace domain::Tags
