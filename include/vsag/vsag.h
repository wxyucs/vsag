#pragma once

#include <string>

namespace vsag {

/**
  * Get the version based on git revision
  * 
  * @return the version text
  */
extern std::string
version();

}  // namespace vsag

#include "binaryset.h"
#include "bitset.h"
#include "constants.h"
#include "dataset.h"
#include "errors.h"
#include "expected.hpp"
#include "factory.h"
#include "index.h"
#include "option.h"
#include "readerset.h"
#include "utils.h"
