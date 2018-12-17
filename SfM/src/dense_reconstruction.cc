// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#include "dense_reconstruction.h"

#include <fstream>
#include <filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "utils/basic_funcs.h"


namespace objectsfm {

	DenseReconstruction::DenseReconstruction()
	{
	}

	DenseReconstruction::~DenseReconstruction()
	{
	}

	void DenseReconstruction::Run(std::string fold)
	{
	}

}  // namespace objectsfm
