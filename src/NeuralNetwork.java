//@author Tengfu Fang and Di Sun



import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;


public class NeuralNetwork
	{
		static int hiddenLayerNum;
		static int inputSampleNum;
		static Double alpha=0.01;//learning speed alpha
		static double B=0.1;
		
		static Double[][] input25=new Double[60000][25];
		static Double[][] hidden_output_weights;
		static Double[] output_threshold=new Double[10];
		static Double[][] inputTest;
		static Double[][] actualOutput=new Double[60000][10];
		static Double[][] expectedOutput=new Double[60000][10];
		static ArrayList<Double[][]> hiddenMinusTohidden_weights=new ArrayList<Double[][]>();
		static ArrayList<Double[]> hidden_thresholds=new ArrayList<Double[]>();
		
		static int[] hiddenNodesNums;
		static Double[] error=new Double[10];

		//input something like NeuralNetwork(4,"100,100,100,50,0,0xxxxx")
		public NeuralNetwork(int hiddenLayerNum,String hiddenLayerNodesNum,int inputSampleNum) throws IOException
			{
				
				this.inputSampleNum=inputSampleNum;
				this.hiddenLayerNum=hiddenLayerNum;
				hiddenNodesNums=new int[hiddenLayerNum];
				determineStructure(hiddenLayerNum,hiddenLayerNodesNum);
				initialize();
				readInput(inputSampleNum);
			}
		
		public void determineStructure(int hiddenLayerNum,String hiddenLayerNodesNum)
			{
				String[] ArrayNodeNum=hiddenLayerNodesNum.split(",");
				for(int i=0;i<hiddenLayerNum;i++)
					{
						hiddenNodesNums[i]=new Integer(ArrayNodeNum[i]);
					}
				Double[][] inputToHiddenWeights=new Double[25][new Integer(ArrayNodeNum[0])];
				hiddenMinusTohidden_weights.add(inputToHiddenWeights);
				Double[] inputToHiddenThreshold=new Double[new Integer(ArrayNodeNum[0])];
				hidden_thresholds.add(inputToHiddenThreshold);
				for(int i=1;i<hiddenLayerNum;i++)
					{
						Double[][] hiddenToHiddenWeights=new Double[new Integer(ArrayNodeNum[i-1])][new Integer(ArrayNodeNum[i])];
						hiddenMinusTohidden_weights.add(hiddenToHiddenWeights);
						Double[] hiddenToHiddenThreshold=new Double[new Integer(ArrayNodeNum[i])];
						hidden_thresholds.add(hiddenToHiddenThreshold);
					}
			}
		public void readInput(int inputSampleNum) throws IOException
			{
				BufferedReader br = new BufferedReader(new FileReader("/Users/Otakuftf/Desktop/train.csv"));
				for(int j=0;j<inputSampleNum;j++)
					{
						String input=br.readLine();
						String[] split=input.split(",");
						for(int i=1;i<26;i++)
							{
								input25[j][i-1]=new Double(split[i])/1000.0;
								//System.out.print(input25[j][i-1]=new Double(split[i])/10.0f);
							}
						//expectedOutput=new Double(split[0]);
						for(int i=0;i<10;i++)
							{
								expectedOutput[j][i]=-1.0;
							}
						Double DoubledExpected=new Double(split[0]);
						
						expectedOutput[j][DoubledExpected.intValue()]=1.0;
					}
				System.err.print("Reading Input finished"+"\n");
			}
		public static void initialize()
			{
				DecimalFormat df = new DecimalFormat("0.00000");
				for(int m=0;m<hiddenLayerNum;m++)
					{
						Double[][] hiddenToHiddenWeight=hiddenMinusTohidden_weights.get(m);
						Double[] hiddenThreshold=hidden_thresholds.get(m);
						for(int j=0;j<hiddenNodesNums[m];j++)
							{
								Double random=(Double) (Math.random()*2*0.01-0.01);
								hiddenThreshold[j]=new Double(df.format(random));
							}
						if(m==0)
							{
								for(int i=0;i<25;i++)
									{
										for(int j=0;j<hiddenNodesNums[m];j++)
											{
												Double random=(Double) (Math.random()*2*0.01-0.01);
												hiddenToHiddenWeight[i][j]=new Double(df.format(random));
											}
									}
							}
						else
							{
								for(int i=0;i<hiddenNodesNums[m-1];i++)
									{
										for(int j=0;j<hiddenNodesNums[m];j++)
											{
												Double random=(Double) (Math.random()*2*0.01-0.01);
												hiddenToHiddenWeight[i][j]=new Double(df.format(random));
											}
									}
							}
					}
			
				hidden_output_weights=new Double[hiddenNodesNums[hiddenLayerNum-1]][10];
				for(int i=0;i<hiddenNodesNums[hiddenLayerNum-1];i++)
					{
						for(int j=0;j<10;j++)
							{
								Double random=(Double) (Math.random()*2*0.01-0.01);
								hidden_output_weights[i][j]=new Double(df.format(random));
							}
					}
			
				for(int i=0;i<10;i++)
					{
						Double random=(Double) (Math.random()*2*0.01-0.01);
						output_threshold[i]=new Double(df.format(random));
					}
				System.err.print("Initializing finished\n");
			}
		public void trainOnce()
			{	
				for(int w=0;w<inputSampleNum;w++)
					{
						ArrayList<Double[]> XHiddenArray=new ArrayList<Double[]>();
						ArrayList<Double[]> YHiddenArray=new ArrayList<Double[]>();
						//loop through hidden layers
						for(int m=0;m<hiddenLayerNum;m++)
							{
								Double[] XForHidden=new Double[hiddenNodesNums[m]];
								Double[] YForHidden=new Double[hiddenNodesNums[m]];
								//calculate XForHidden and YForHidden and store
								Double[][] hidden_hidden_weights=hiddenMinusTohidden_weights.get(m);
								Double[] hidden_threshold=hidden_thresholds.get(m);
								for(int i=0;i<hiddenNodesNums[m];i++)
									{
										XForHidden[i]=0.0;
										try
											{
												Double[] YForPrevHidden=YHiddenArray.get(m-1);
										for(int j=0;j<hiddenNodesNums[m-1];j++)
											{
												XForHidden[i]+=YForPrevHidden[j]*hidden_hidden_weights[j][i];
											}
											}
										catch (Exception e)
											{
												for(int j=0;j<25;j++)
													{
														XForHidden[i]+=input25[w][j]*hidden_hidden_weights[j][i];
													}
											}
										XForHidden[i]=XForHidden[i]+hidden_threshold[i];
										YForHidden[i]=ActivationFunction(XForHidden[i]);
									}
								XHiddenArray.add(XForHidden);
								YHiddenArray.add(YForHidden);
							}
						//calculate X/Y for output and error & gradient.
						Double XForOutput[]=new Double[10];
						Double outputGradient[]=new Double[10];
						Double[] YForHidden=YHiddenArray.get(hiddenLayerNum-1);
						for(int j=0;j<10;j++)
							{
								XForOutput[j]=0.0;
								
							for(int i=0;i<hiddenNodesNums[hiddenLayerNum-1];i++)
								{
									XForOutput[j]+=YForHidden[i]*hidden_output_weights[i][j];
								}
							XForOutput[j]=XForOutput[j]-output_threshold[j];
							actualOutput[w][j]=ActivationFunction(XForOutput[j]);
							error[j]=actualOutput[w][j]-expectedOutput[w][j];
							
							//Produce gradients for hidden/output and adjusted w.
							//Possible to remove momentum
							outputGradient[j]=error[j];
							//outputGradient[j]+=B*outputGradient[j];
							}
						//update hidden_output weights and threshold
						Double[] YHidden=YHiddenArray.get(hiddenLayerNum-1);
						for(int j=0;j<10;j++)
							{
							for(int i=0;i<hiddenNodesNums[hiddenLayerNum-1];i++)
								{
									hidden_output_weights[i][j]-=alpha*YHidden[i]*outputGradient[j];//+B*wH_O[i][j];
									//wH_O[i][j]=alpha*actualHiddenOutput[i]*outputGradient[j]+B*wH_O[i][j];
								}
							output_threshold[j]+=alpha*(-1)*outputGradient[j];
							}
						//calculate hidden gradient.
						ArrayList<Double[]> hiddenGradients=new ArrayList<Double[]>();
						for(int m=hiddenLayerNum-1;m>=0;m--)
							{
								Double[] hiddenGradient=new Double[hiddenNodesNums[m]];
								Double[][] hidden_hidden_weights=hiddenMinusTohidden_weights.get(m);
								Double[] hiddenThreshold=hidden_thresholds.get(m);
								if(m==hiddenLayerNum-1)
									{
										Double[] XHidden=XHiddenArray.get(m);
										for(int i=0;i<hiddenNodesNums[m];i++)
											{
												Double sum=0.0;
												for(int j=0;j<10;j++)
													{
														sum+=outputGradient[j]*hidden_output_weights[i][j];
													}
												hiddenGradient[i]=DerivativeActive(XHidden[i])*sum;
												//hiddenGradient[i]+=B*hiddenGradient[i];
											}
									}
								else
									{
										Double[] nextHiddenGradient=hiddenGradients.get(hiddenLayerNum-m-2);
										Double[] XHidden=XHiddenArray.get(m);
										Double[][] nextHidden_hidden_weights=hiddenMinusTohidden_weights.get(m+1);
										for(int i=0;i<hiddenNodesNums[m];i++)
											{
												Double sum=0.0;
												for(int j=0;j<hiddenNodesNums[m+1];j++)
													{
														sum+=nextHiddenGradient[j]*nextHidden_hidden_weights[i][j];
													}
												hiddenGradient[i]=DerivativeActive(XHidden[i])*sum;
												//hiddenGradient[i]+=B*hiddenGradient[i];
											}
									}
								//update hidden weights & threshold
								if(m==0)
									{
										for(int i=0;i<hiddenNodesNums[m];i++)
											{
											for(int j=0;j<25;j++)
												{
													hidden_hidden_weights[j][i]-=alpha*input25[w][j]*hiddenGradient[i];
												}
											hiddenThreshold[i]+=alpha*(-1)*hiddenGradient[i];
											}
									}
								else
									{
										Double[] PreYHidden=YHiddenArray.get(m-1);
										for(int i=0;i<hiddenNodesNums[m];i++)
											{
											for(int j=0;j<hiddenNodesNums[m-1];j++)
												{
													hidden_hidden_weights[j][i]-=alpha*PreYHidden[j]*hiddenGradient[i];
												}
											hiddenThreshold[i]+=alpha*(-1)*hiddenGradient[i];
											}
									}
								hiddenGradients.add(hiddenGradient);
							}
					}
			}
		public Double train(int times)
			{
			    double a=0;
				for(int i=0;i<times;i++)
					{
						trainOnce();
						a=checkToStop();
						System.out.print("Successful count rate= "+a+"\n");
						if(a>=0.815)
							{
							    i=i+1;
								System.err.print("Training finished with >0.85 accuracy and with "+i+" loops");
								break;
							}
					}
				
				return a;
			}
		private static double checkToStop()
			{
				double count=0;
				for(int i=0;i<inputSampleNum;i++)
					{
						int largestValueColumn=findLargestColumn(actualOutput[i]);
						if(largestValueColumn!=25&&expectedOutput[i][largestValueColumn]==1.0f)
							{
								count+=1;
							}
					}
						return count/inputSampleNum;
			}
		//Only for stage1 prediction
		public static void prediction() throws IOException
		{
			inputTest=new Double[10000][25];
			BufferedReader br = new BufferedReader(new FileReader("/Users/Otakuftf/Desktop/test-nolabel.csv"));
			for(int j=0;j<10000;j++)
				{
					String input=br.readLine();
					String[] split=input.split(" ");
					for(int i=1;i<26;i++)
						{
							inputTest[j][i-1]=new Double(split[i])/1000.0;
						}
				}
			Double[][] input_hidden_weights=hiddenMinusTohidden_weights.get(0);
			Double[] hidden_threshold=hidden_thresholds.get(0);
			Double[] actualHiddenOutput=new Double[100];
			Double[][] prediction=new Double[10000][10];
			for(int w=0;w<10000;w++)
			{
		
			//forward feeding
			//update actualOutput in hidden layer
			Double[] XForHidden=new Double[100];
			for(int i=0;i<100;i++)
				{
					XForHidden[i]=0.0;
					for(int j=0;j<25;j++)
						{
							XForHidden[i]+=inputTest[w][j]*input_hidden_weights[j][i];
						}
					XForHidden[i]=XForHidden[i]+hidden_threshold[i];
					actualHiddenOutput[i]=ActivationFunction(XForHidden[i]);
				}
			//produce actualOutput
			Double XForOutput[]=new Double[10];
		for(int j=0;j<10;j++)
			{
				XForOutput[j]=0.0;
			for(int i=0;i<100;i++)
				{
					XForOutput[j]+=actualHiddenOutput[i]*hidden_output_weights[i][j];
				}
			XForOutput[j]=XForOutput[j]-output_threshold[j];
			prediction[w][j]=ActivationFunction(XForOutput[j]);
			}
			}
			
			//print the prediction result
			String result = "";
			for(int i=0;i<10000;i++)
			{
				   int predictionValue;
				   predictionValue=findLargestColumn(prediction[i]);
				   if(predictionValue==25)
						result+="9\n";
				   else
						result+=predictionValue+".0\n";
				   
			}
			PrintWriter out = new PrintWriter("/Users/Otakuftf/Desktop/result.txt");
			out.print(result);
			out.close();
		}
		public static Double ActivationFunction(Double x)
			{
				//tanh z=sinh z/cosh z=(e 2z -1)/(e 2z +1)
				//System.err.print(2*x+"\n");
				//System.out.print((Math.pow(Math.E, 2*x))+"\n\n");
				//Double newD=Math.pow(Math.E, 2*x);
				if(x>150.0)
					return 1.0;
				else if(x<-150.0)
					return -1.0;
						Float fx=x.floatValue();
				return new Double((Math.pow(Math.E, 2*fx)-1)/(Math.pow(Math.E, 2*fx)+1));
			}
		public static Double DerivativeActive(Double x)
			{
				//square of sech(x)
				Float fx=x.floatValue();
				return new Double(((2/(Math.pow(Math.E, fx)+Math.pow(Math.E, -fx)))*(2/(Math.pow(Math.E, fx)+Math.pow(Math.E, -fx)))
));				
			}
		public static int findLargestColumn(Double[] input)
			{
				Double largest=-1.0;
				for(int i=0;i<input.length;i++)
					{
						if(input[i]>largest)
							{
								largest=input[i];
							}
					}
				if(largest==-1.0f)
					{
						return 25;
					}
				else
					{
						for(int i=0;i<input.length;i++)
							{
								if(input[i]==largest)
									{
										return i;
									}
							}
					}
				return 30;
			}
	}
