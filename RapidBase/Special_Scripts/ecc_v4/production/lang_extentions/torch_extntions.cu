#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#define ADD_TEMPLATE_PARM( OBJ, PARAM) OBJ<PARAM>

