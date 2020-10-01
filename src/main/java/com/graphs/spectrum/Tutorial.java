package com.graphs.spectrum;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;


public class Tutorial {

	public static void main(String[] args) {
		double[][] arr = {{3, -1, 0, 0, 0, 0, -1, -1, 0, 0},
				{-1, 2, -1, 0, 0, 0, 0, 0, 0, 0},
				{0, -1, 2, -1, 0, 0, 0, 0, 0, 0},
				{0, 0, -1, 2, -1, 0, 0, 0, 0, 0},
				{0, 0, 0, -1, 2, -1, 0, 0, 0, 0},
				{0, 0, 0, 0, -1, 2, -1, 0, 0, 0},
				{-1, 0, 0, 0, 0, -1, 2, 0, 0, 0},
				{-1, 0, 0, 0, 0, 0, 0, 3, -1, -1},
				{0, 0, 0, 0, 0, 0, 0, -1, 2, -1},
				{0, 0, 0, 0, 0, 0, 0, -1, -1, 2}};
		INDArray x_2d = Nd4j.create(arr);
		
		int size = (int) x_2d.transpose().size(1);
		
//		System.out.println(x_2d.getRow(1));
//		
//		INDArray ze = Nd4j.zeros(3,5,6);
//		
//		System.out.println(ze.rank());
		
		INDArray eigens = Eigen.symmetricGeneralizedEigenvalues(x_2d);
		System.out.println(eigens.getRow(0));
//		for(int i = 0; i < 10; i++) {
//			for(int j = 0; j < 10; j++) {
//				System.out.println(eigens.getDouble(i, j));
//			}
//		}

	}

}
