// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"

#include <cstddef>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <memory>
#include <pup.h>

#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/MakeString.hpp"

namespace integration::detail {

GslQuadAdaptiveImpl::GslQuadAdaptiveImpl(const size_t max_intervals) noexcept
    : max_intervals_(max_intervals), gsl_integrand_(gsl_function{}) {
  initialize();
}

void GslQuadAdaptiveImpl::pup(PUP::er& p) noexcept {
  p | max_intervals_;
  if (p.isUnpacking()) {
    initialize();
  }
}

void GslQuadAdaptiveImpl::GslIntegrationWorkspaceDeleter::operator()(
    gsl_integration_workspace* workspace) const noexcept {
  gsl_integration_workspace_free(workspace);
}

void GslQuadAdaptiveImpl::initialize() noexcept {
  workspace_ = std::unique_ptr<gsl_integration_workspace,
                               GslIntegrationWorkspaceDeleter>{
      gsl_integration_workspace_alloc(max_intervals_)};
}

void check_status_code(const int status_code) {
  if (status_code != 0) {
    throw convergence_error(MakeString{}
                            << "Integration failed with GSL error: "
                            << gsl_strerror(status_code));
  }
}

void disable_gsl_error_handling() noexcept { gsl_set_error_handler_off(); }

}  // namespace integration::detail
