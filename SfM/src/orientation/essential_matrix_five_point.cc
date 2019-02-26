
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

#include "orientation/essential_matrix_five_point.h"

#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Dense>

#include "utils/basic_funcs.h"

typedef Eigen::Matrix<double, 10, 10> Matrix10d;

namespace objectsfm {

	bool EssentialMatrixFivePoints::FivePointEssentialMatrixRANSAC(const std::vector<Eigen::Vector2d>& image1_points, 
		const std::vector<Eigen::Vector2d>& image2_points, Eigen::Matrix3d &E)
	{
		int num = image1_points.size();

		std::vector<Eigen::Matrix3d> essential_matrices;
		if (num<5)
		{
			return false;
		}
		else if (num<10)
		{
			// inadequate points, use all for Essential Matrix estimation
			if (!FivePointEssentialMatrix(image1_points, image2_points, &essential_matrices))
			{
				return false;
			}
		}
		else
		{
			int ransac_times = 100;
			for (int i=0; i<ransac_times; ++i)
			{
				// get random index
				std::vector<int> idxs;
				math::RandVectorN(0, num, 5, idxs);

				std::vector<Eigen::Vector2d> image1_points_partial, image2_points_partial;
				for (int j = 0; j<idxs.size(); ++j)
				{
					int idx = idxs[j];
					image1_points_partial.push_back(image1_points[idx]);
					image2_points_partial.push_back(image2_points[idx]);
				}

				if (!FivePointEssentialMatrix(image1_points_partial, image2_points_partial, &essential_matrices))
				{
					continue;
				}
			}
		}

		if (essential_matrices.size()<4)
		{
			return false;
		}

		// find the one with least error
		double error_min = 1000000.0;
		int idx_min = 0;
		for (int i=0; i<essential_matrices.size();++i)
		{
			double error_i = Error(essential_matrices[i], image1_points, image2_points);
			if (error_i < error_min)
			{
				error_min = error_i;
				idx_min = i;
			}
		}

		E = essential_matrices[idx_min];

		return true;
	}


	// Implementation of Nister from "An Efficient Solution to the Five-Point Relative Pose Problem"
	// image1_points = [x,y,1], [x,y,1] = [u,v,f]/f, (u,v) is the centralized pixel coordinate
	bool EssentialMatrixFivePoints::FivePointEssentialMatrix(const std::vector<Eigen::Vector2d>& image1_points,
		const std::vector<Eigen::Vector2d>& image2_points,
		std::vector<Eigen::Matrix3d>* essential_matrices)
	{
		// Step 1. Create the nx9 matrix containing epipolar constraints.
		//   Essential matrix is a linear combination of the 4 vectors spanning the
		//   null space of this matrix.
		Eigen::MatrixXd epipolar_constraint(image1_points.size(), 9);
		for (int i = 0; i < image1_points.size(); i++)
		{
			// Fill matrix with the epipolar constraint from q'_t*E*q = 0. Where q is
			// from the first image, and q' is from the second.
			epipolar_constraint.row(i) <<
				image2_points[i].x() * image1_points[i].x(),
				image2_points[i].y() * image1_points[i].x(),
				image1_points[i].x(),
				image2_points[i].x() * image1_points[i].y(),
				image2_points[i].y() * image1_points[i].y(),
				image1_points[i].y(),
				image2_points[i].x(),
				image2_points[i].y(),
				1.0;
		}

		Eigen::Matrix<double, 9, 4> null_space;

		// Extract the null space from a minimal sampling (using LU) or non-minimal
		// sampling (using SVD).
		if (image1_points.size() == 5)
		{
			const Eigen::FullPivLU<Eigen::MatrixXd> lu(epipolar_constraint);
			if (lu.dimensionOfKernel() != 4)
			{
				return false;
			}
			null_space = lu.kernel();
		}
		else
		{
			const Eigen::JacobiSVD<Eigen::MatrixXd> svd(epipolar_constraint.transpose() * epipolar_constraint, Eigen::ComputeFullV);
			null_space = svd.matrixV().rightCols<4>();
		}

		const Eigen::Matrix<double, 1, 4> null_space_matrix[3][3] =
		{
			{ null_space.row(0), null_space.row(3), null_space.row(6) },
			{ null_space.row(1), null_space.row(4), null_space.row(7) },
			{ null_space.row(2), null_space.row(5), null_space.row(8) }
		};

		// Step 2. Expansion of the epipolar constraints on the determinant and trace.
		const Eigen::Matrix<double, 10, 20> constraint_matrix = BuildConstraintMatrix(null_space_matrix);

		// Step 3. Eliminate part of the matrix to isolate polynomials in z.
		Eigen::FullPivLU<Matrix10d> c_lu(constraint_matrix.block<10, 10>(0, 0));
		Matrix10d eliminated_matrix = c_lu.solve(constraint_matrix.block<10, 10>(0, 10));

		Matrix10d action_matrix = Matrix10d::Zero();
		action_matrix.block<3, 10>(0, 0) = eliminated_matrix.block<3, 10>(0, 0);
		action_matrix.row(3) = eliminated_matrix.row(4);
		action_matrix.row(4) = eliminated_matrix.row(5);
		action_matrix.row(5) = eliminated_matrix.row(7);
		action_matrix(6, 0) = -1.0;
		action_matrix(7, 1) = -1.0;
		action_matrix(8, 3) = -1.0;
		action_matrix(9, 6) = -1.0;

		Eigen::EigenSolver<Matrix10d> eigensolver(action_matrix);
		const auto& eigenvectors = eigensolver.eigenvectors();
		const auto& eigenvalues = eigensolver.eigenvalues();

		// Now that we have x, y, and z we need to substitute them back into the null
		// space to get a valid essential matrix solution.
		for (int i = 0; i < 10; i++)
		{
			// Only consider real solutions.
			if (eigenvalues(i).imag() != 0)
			{
				continue;
			}
			Eigen::Matrix3d ematrix;
			Eigen::Map<Eigen::Matrix<double, 9, 1> >(ematrix.data()) = null_space * eigenvectors.col(i).tail<4>().real();
			essential_matrices->emplace_back(ematrix);
		}

		return essential_matrices->size() > 0;
	}

	// Multiply two degree one polynomials of variables x, y, z.
	// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
	// Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
	Eigen::Matrix<double, 1, 10> EssentialMatrixFivePoints::MultiplyDegOnePoly(const Eigen::RowVector4d& a, const Eigen::RowVector4d& b)
	{
		Eigen::Matrix<double, 1, 10> output;
		// x^2
		output(0) = a(0) * b(0);
		// xy
		output(1) = a(0) * b(1) + a(1) * b(0);
		// y^2
		output(2) = a(1) * b(1);
		// xz
		output(3) = a(0) * b(2) + a(2) * b(0);
		// yz
		output(4) = a(1) * b(2) + a(2) * b(1);
		// z^2
		output(5) = a(2) * b(2);
		// x
		output(6) = a(0) * b(3) + a(3) * b(0);
		// y
		output(7) = a(1) * b(3) + a(3) * b(1);
		// z
		output(8) = a(2) * b(3) + a(3) * b(2);
		// 1
		output(9) = a(3) * b(3);
		return output;
	}

	// Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
	// x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
	Eigen::Matrix<double, 1, 20> EssentialMatrixFivePoints::MultiplyDegTwoDegOnePoly(const Eigen::Matrix<double, 1, 10>& a, const Eigen::RowVector4d& b)
	{
		Eigen::Matrix<double, 1, 20> output;
		// x^3
		output(0) = a(0) * b(0);
		// x^2y
		output(1) = a(0) * b(1) + a(1) * b(0);
		// xy^2
		output(2) = a(1) * b(1) + a(2) * b(0);
		// y^3
		output(3) = a(2) * b(1);
		// x^2z
		output(4) = a(0) * b(2) + a(3) * b(0);
		// xyz
		output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
		// y^2z
		output(6) = a(2) * b(2) + a(4) * b(1);
		// xz^2
		output(7) = a(3) * b(2) + a(5) * b(0);
		// yz^2
		output(8) = a(4) * b(2) + a(5) * b(1);
		// z^3
		output(9) = a(5) * b(2);
		// x^2
		output(10) = a(0) * b(3) + a(6) * b(0);
		// xy
		output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
		// y^2
		output(12) = a(2) * b(3) + a(7) * b(1);
		// xz
		output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
		// yz
		output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
		// z^2
		output(15) = a(5) * b(3) + a(8) * b(2);
		// x
		output(16) = a(6) * b(3) + a(9) * b(0);
		// y
		output(17) = a(7) * b(3) + a(9) * b(1);
		// z
		output(18) = a(8) * b(3) + a(9) * b(2);
		// 1
		output(19) = a(9) * b(3);
		return output;
	}

	// Shorthand for multiplying the Essential matrix with its transpose.
	Eigen::Matrix<double, 1, 10> EssentialMatrixFivePoints::EETranspose(const Eigen::Matrix<double, 1, 4> null_space[3][3], int i, int j)
	{
		return MultiplyDegOnePoly(null_space[i][0], null_space[j][0]) +
			MultiplyDegOnePoly(null_space[i][1], null_space[j][1]) +
			MultiplyDegOnePoly(null_space[i][2], null_space[j][2]);
	}

	// Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
	Eigen::Matrix<double, 9, 20> EssentialMatrixFivePoints::GetTraceConstraint(const Eigen::Matrix<double, 1, 4> null_space[3][3])
	{
		Eigen::Matrix<double, 9, 20> trace_constraint;

		// Comput EEt.
		Eigen::Matrix<double, 1, 10> eet[3][3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				eet[i][j] = 2 * EETranspose(null_space, i, j);
			}
		}

		// Compute the trace.
		const Eigen::Matrix<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];

		// Multiply EEt with E.
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				trace_constraint.row(3 * i + j) =
					MultiplyDegTwoDegOnePoly(eet[i][0], null_space[0][j]) +
					MultiplyDegTwoDegOnePoly(eet[i][1], null_space[1][j]) +
					MultiplyDegTwoDegOnePoly(eet[i][2], null_space[2][j]) -
					0.5 * MultiplyDegTwoDegOnePoly(trace, null_space[i][j]);
			}
		}

		return trace_constraint;
	}

	Eigen::Matrix<double, 1, 20> EssentialMatrixFivePoints::GetDeterminantConstraint(const Eigen::Matrix<double, 1, 4> null_space[3][3])
	{
		// Singularity constraint.
		const Eigen::Matrix<double, 1, 20> determinant =
			MultiplyDegTwoDegOnePoly(
				MultiplyDegOnePoly(null_space[0][1], null_space[1][2]) -
				MultiplyDegOnePoly(null_space[0][2], null_space[1][1]),
				null_space[2][0]) +
			MultiplyDegTwoDegOnePoly(
				MultiplyDegOnePoly(null_space[0][2], null_space[1][0]) -
				MultiplyDegOnePoly(null_space[0][0], null_space[1][2]),
				null_space[2][1]) +
			MultiplyDegTwoDegOnePoly(
				MultiplyDegOnePoly(null_space[0][0], null_space[1][1]) -
				MultiplyDegOnePoly(null_space[0][1], null_space[1][0]),
				null_space[2][2]);
		return determinant;
	}

	Eigen::Matrix<double, 10, 20> EssentialMatrixFivePoints::BuildConstraintMatrix(const Eigen::Matrix<double, 1, 4> null_space[3][3])
	{
		Eigen::Matrix<double, 10, 20> constraint_matrix;
		constraint_matrix.block<9, 20>(0, 0) = GetTraceConstraint(null_space);
		constraint_matrix.row(9) = GetDeterminantConstraint(null_space);
		return constraint_matrix;
	}

	double EssentialMatrixFivePoints::Error(const Eigen::Matrix3d& E, const std::vector<Eigen::Vector2d>& image1_points,
		const std::vector<Eigen::Vector2d>& image2_points)
	{
		double total_error = 0.0;
		for (int i=0; i<image1_points.size(); ++i)
		{
			const Eigen::Vector3d epiline_x = E * image1_points[i].homogeneous();
			const double numerator_sqrt = image2_points[i].homogeneous().dot(epiline_x);
			const Eigen::Vector4d denominator(image2_points[i].homogeneous().dot(E.col(0)), 
				image2_points[i].homogeneous().dot(E.col(1)), 
				epiline_x[0], epiline_x[1]);

			// Finally, return the complete Sampson distance.
			total_error += numerator_sqrt * numerator_sqrt / denominator.squaredNorm();
		}
		
		return total_error;
	}

}// namespace objectsfm
