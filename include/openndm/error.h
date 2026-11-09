//! \file error.h
//! Typed error handling for the OpenNDM core (FR-OPT-6).

#pragma once

#include <stdexcept>
#include <string>

namespace openndm {

//! Base class for every exception thrown by the OpenNDM core.
//!
//! The Python bindings translate each of these to a distinct Python exception
//! type so that a caller can react to a non-converged solve without parsing
//! strings and without the process ever aborting (FR-OPT-6).
class Error : public std::runtime_error {
public:
  //! \param what message, carried unchanged into the Python exception.
  explicit Error(const std::string& what)
      : std::runtime_error(what)
  {
  }
};

//! Invalid or inconsistent user input (geometry, settings, library).
class InputError : public Error {
public:
  //! \param what message naming the input that is wrong and why.
  explicit InputError(const std::string& what)
      : Error(what)
  {
  }
};

//! A cross section library failed one of the FR-XS-8 validation checks.
class LibraryError : public Error {
public:
  //! \param what message naming the check the library failed.
  explicit LibraryError(const std::string& what)
      : Error(what)
  {
  }
};

//! An iterative solve exhausted its iteration budget without converging.
class ConvergenceError : public Error {
public:
  //! \param what message naming the solve that gave up.
  //! \param iterations iterations taken before giving up.
  //! \param residual the residual it had reached, in the solve's own measure.
  ConvergenceError(const std::string& what, int iterations, double residual)
      : Error(what),
        iterations_(iterations),
        residual_(residual)
  {
  }

  //! Iterations the solve had taken when it gave up.
  int iterations() const { return iterations_; }
  //! Residual it had reached, in the measure the solve converges on.
  double residual() const { return residual_; }

private:
  int iterations_;
  double residual_;
};

//! A requested capability exists in the API but is not implemented yet.
class NotImplementedError : public Error {
public:
  //! \param what message naming the capability that is not implemented.
  explicit NotImplementedError(const std::string& what)
      : Error(what)
  {
  }
};

}
