#include "openndm/kernel.h"

#include "openndm/error.h"

namespace openndm {

std::unique_ptr<Kernel> make_fdm_kernel();
std::unique_ptr<Kernel> make_nem_kernel();
std::unique_ptr<Kernel> make_sanm_kernel();

std::unique_ptr<Kernel> Kernel::create(KernelType type)
{
  switch (type) {
  case KernelType::fdm:
    return make_fdm_kernel();
  case KernelType::nem:
    return make_nem_kernel();
  case KernelType::sanm:
    return make_sanm_kernel();
  }
  throw InputError("unknown kernel type");
}

} // namespace openndm
