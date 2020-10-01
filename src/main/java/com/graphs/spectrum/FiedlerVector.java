package com.graphs.spectrum;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

public class FiedlerVector {
	
	public static INDArray findFiedlerVector(INDArray laplacianMatrix) {
		
		INDArray eigenValue = Eigen.symmetricGeneralizedEigenvalues(laplacianMatrix);
				
		return Nd4j.sort(eigenValue, 0, false);							
	}
}
