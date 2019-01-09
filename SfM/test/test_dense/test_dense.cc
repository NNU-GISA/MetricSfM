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

#include <iostream>
#include <fstream>
#include "basic_structs.h"
#include "dense_reconstruction.h"

void main(void)
{
	objectsfm::DenseReconstruction denser;

	std::string fold = "F:\\Database\\GoPro\\11-20\\1114_HERO7_data\\front\\GX020005";
	denser.Run(fold);
	//denser.Depth2Points(fold + "\\dense_elas_result");
}