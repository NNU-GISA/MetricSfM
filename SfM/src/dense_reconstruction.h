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

#ifndef OBJECTSFM_SYSTEM_DENSE_H_
#define OBJECTSFM_SYSTEM_DENSE_H_

#include <opencv2/opencv.hpp>

#include "basic_structs.h"
#include "database.h"
#include "camera.h"
#include "structure.h"
#include "optimizer.h"


namespace objectsfm {


class DenseReconstruction
{
public:
	DenseReconstruction();
	~DenseReconstruction();

	DenseOptions options_;

	// the main pipeline
	void Run(std::string fold);

	void ReadinPoseFile(std::string sfm_file);

	void SGMDense();

	void ELASDense();

	void EpipolarRectification(cv::Mat &img1, cv::Mat K1, cv::Mat R1, cv::Mat t1,
		                       cv::Mat &img2, cv::Mat K2, cv::Mat R2, cv::Mat t2, 
		                       bool write_out, cv::Mat &R1_, cv::Mat &R2_);

	void SavePoseFile(std::string sfm_file);

private:
	int cols, rows;
	std::string fold, fold_img, fold_output;
	std::vector<std::string> names;
	std::vector<cv::Mat> Ks, Rs, ts, Rs_new, ts_new;
};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_DENSE_H_
