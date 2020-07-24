// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/IdentityMatrix.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"
#include "Utilities/TMPL.hpp"
namespace {
struct ScalarFieldTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace LinearSolver::Schwarz {

SPECTRE_TEST_CASE(
    "Unit.ParallelSchwarz.SubdomainPreconditioners.ExplicitInverse",
    "[Unit][ParallelAlgorithms][LinearSolver]") {
  using SubdomainData =
      ElementCenteredSubdomainData<1, tmpl::list<ScalarFieldTag>>;
  SubdomainData used_for_size{3};
  used_for_size.overlap_data.emplace(
      OverlapId<1>{Direction<1>::lower_xi(), ElementId<1>{0}},
      typename SubdomainData::OverlapData{2});
  const ExplicitInverse<1> preconditioner{
      [](const SubdomainData& arg) noexcept { return arg; }, used_for_size};
  CHECK(preconditioner.size() == 5);
  CHECK_MATRIX_APPROX(preconditioner.matrix_representation(),
                      blaze::IdentityMatrix<double>(5));
}

}  // namespace LinearSolver::Schwarz
