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

#ifndef OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_
#define OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_

#include "basic_structs.h"
#include "graph.h"
#include "database.h"
#include "camera.h"
#include "structure.h"
#include "optimizer.h"

namespace objectsfm {

// the system is designed to handle sfm from internet images, in which not
// all the focal lengths are known
class IncrementalSfM
{
public:
	IncrementalSfM();
	~IncrementalSfM();

	IncrementalSfMOptions options_;

	// initialize
	void InitializeSystem();

	// the main pipeline
	void Run();

	// find initial seed pair for 3D reconstruction
	bool FindSeedPairThenReconstruct();

	// find and sort the images according to their 2d-3d correspondences
	void FindImageToLocalize(std::vector<int> &image_ids, std::vector<std::vector<std::pair<int, int>>> &corres_2d3d,
		std::vector<std::vector<int>> &visible_cams);

	bool LocalizeImage(int id_img, std::vector<std::pair<int, int>> &corres_2d3d, std::vector<int> &visible_cams);

	void GenerateNew3DPoints();

	void PartialBundleAdjustment(int idx);

	void FullBundleAdjustment();

	void SingleBundleAdjustment(int idx);

	void SaveModel();

	void WriteCameraPointsOut(std::string path);

	void WriteTempResultOut(std::string path);

	void ReadTempResultIn(std::string path);

private:
	bool CameraAssociateCameraModel(Camera* cam);

	void SortImagePairs(std::vector<std::pair<int, int>> &seed_pair_hyps);

	void RemovePointOutliers();

	// immutable all the cameras and points for the next partial bundle adjustment 
	void ImmutableCamsPoints();

	// mutable all the cameras and points for the full bundle adjustment 
	void MutableCamsPoints();

	void UpdateVisibleGraph(int idx_new_cam, std::vector<int> idxs_visible_cam);

private:
	BundleAdjustOptions bundle_full_options_;
	BundleAdjustOptions bundle_partial_options_;

	Database db_;  // data base
	Graph graph_;  // graph
	std::vector<CameraModel*> cam_models_; // camera models
	std::vector<Camera*> cams_;  // cameras
	std::vector<Point3D*> pts_;  // structures
	std::map<int, int> img_cam_map_; // first, image id; second, corresponding camera id
	std::vector<bool> is_img_processed_;
	std::vector<int> localize_fail_times_;
	bool found_seed_;
	int num_reconstructions_;
};

}  // namespace objectsfm

#endif  // OBJECTSFM_SYSTEM_INCREMENTAL_SFM_SYSTEM_H_
