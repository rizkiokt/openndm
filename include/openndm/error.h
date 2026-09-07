//! \file error.h
//! Typed error handling for the OpenNDM core (FR-OPT-6).

#ifndef OPENNDM_ERROR_H
#define OPENNDM_ERROR_H

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
  explicit Error(const std::string& what)
      : std::runtime_error(what)
  {
  }
};

//! Invalid or inconsistent user input (geometry, settings, library).
class InputError : public Error {
public:
  explicit InputError(const std::string& what)
      : Error(what)
  {
  }
};

//! A cross section library failed one of the FR-XS-8 validation checks.
class LibraryError : public Error {
public:
  explicit LibraryError(const std::string& what)
      : Error(what)
  {
  }
};

//! An iterative solve exhausted its iteration budget without converging.
class ConvergenceError : public Error {
public:
  ConvergenceError(const std::string& what, int iterations, double residual)
      : Error(what),
        iterations_(iterations),
        residual_(residual)
  {
  }

  int iterations() const { return iterations_; }
  double residual() const { return residual_; }

private:
  int iterations_;
  double residual_;
};

//! A requested capability exists in the API but is not implemented yet.
class NotImplementedError : public Error {
public:
  explicit NotImplementedError(const std::string& what)
      : Error(what)
  {
  }
};

}  // namespace openndm

#endif  // OPENNDM_ERROR_H
