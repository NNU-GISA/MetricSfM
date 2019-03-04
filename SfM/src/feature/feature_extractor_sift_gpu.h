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

#ifndef OBJECTSFM_FEATURE_EXTRACTOR_SIFTGPU_H_
#define OBJECTSFM_FEATURE_EXTRACTOR_SIFTGPU_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "basic_structs.h"

namespace objectsfm {

	class SIFTGPUExtractor
	{
	public:
		SIFTGPUExtractor();
		~SIFTGPUExtractor();

		static void Run(std::string img_path, ListKeyPoint* keypoints, cv::Mat* descriptors);

		static void Run(cv::Mat &image, ListKeyPoint* keypoints, cv::Mat* descriptors);
	};
}
#endif // OBJECTSFM_FEATURE_EXTRACTOR_CUDASIFT_H_