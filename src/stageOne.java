//@author Tengfu Fang and Di Sun
//this class is used for creating the prediction results for the test set 
//samples.


import java.io.IOException;

public class stageOne {

	public static void main(String[] args) throws IOException{
		int layerNum=1;
		String nodes="100";
		NeuralNetwork net=new NeuralNetwork(layerNum,nodes,60000);
		net.train(50);
		net.prediction();
	}
}