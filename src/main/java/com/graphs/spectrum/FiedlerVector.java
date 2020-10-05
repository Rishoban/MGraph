package com.graphs.spectrum;

import java.util.Random;
import java.util.Vector;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

public class FiedlerVector {
	
	public static INDArray findFiedlerVector(INDArray laplacianMatrix) {
		
		INDArray eigenValue = Eigen.symmetricGeneralizedEigenvalues(laplacianMatrix);
				
		return Nd4j.sort(eigenValue, 0, false);							
	}
	
	public static void lancozsAlgorithm(RealMatrix A) {
		
		Vector<Double> b = new Vector<Double>();
		
		RealMatrix v = new Array2DRowRealMatrix(A.getRowDimension(), A.getColumnDimension());
		
		RealVector r = randomVector(A.getColumnDimension());
		
		Vector<Double> a = new Vector<Double>();
		
		b.add(r.getNorm());
		
		int i = 0;
		while (i != A.getColumnDimension()) {
			i += 1;
			v.setRowVector(i-1, r.mapDivide(b.get(i-1)));
			r = A.operate(v.getRowVector(i-1));
			r =  r.add(v.getRowVector(i-1).mapMultiply(b.get(i-1)).mapMultiply(-1));
			double h = v.getRowVector(i-1).dotProduct(r);
			a.add(h);
			b.add(r.getNorm());
		}		
	}
	
	public static RealVector randomVector(int length) {
		Random ranNumbers = new Random();
		double[] arr = new double[length];
		for(int i = 0; i < length; i++) {
			arr[i] = ranNumbers.nextDouble();
		}
		
		return MatrixUtils.createRealVector(arr);
	}
	
}
