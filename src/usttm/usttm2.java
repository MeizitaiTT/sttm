package usttm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;



public class usttm2 {
	int K;
	int S ;//潜在时间间隔数
	int R;//潜在区域数量
	
	double alpha = 0.01;
	double gamma = 0.01;
	double beta = 0.01;
	
	int iterNum = 100;
	int U,I;
	
	int[][] Zu; //每个用户的主题
	int[][] Ru; //每个用户每次签到的区域
	int[] Reg;//每个区域的签到数量
	int[][] Su; //每个用户的时间间隔
	
	int [][][] nusz;
	int [][][] nusr;
	int [] nsi; //属于时间段的POI个数
	int [][] nzv; //每个区域属于z的POI个数


	double [][] miu;
	double [][] sigma;
	
	double[] miuT;
	double [] sigmaT;
	
	int [][] phisum; //对应nusz;
	int [][] thetasum; //对应nusr;
	int [] nzvsum;//对应于ncz

    
	Train train;
	Train test;
	Map<Integer,List<Integer>> trainmap;
	Map<Integer,List<Integer>> testmap;
//	Map<Integer,List<Double>> test;

	
	double[][] geo; //每个位置的经纬度
	
	
	Input input;
	public usttm2(Input input,int K,int R, int S) throws IOException, ParseException{
		this.K = K;
		this.R = R;
		this.S = S;
		this.input = input;
		this.alpha=50.0/K;
		this.beta=50.0/R;
	    geo =Dataset.readGeo(input.geofile);
	    train = Dataset.readSamples(input.trainfile,input.geofile);
	    trainmap = Dataset.readTest(input.trainfile);
		U = Dataset.U;
		I = Dataset.V;

        
		Zu = new int[U][];
		Ru = new int[U][];
		Reg= new int[R];
		Su = new int[U][];
		
	    nusz = new int[U][S][K];
		nusr = new int[U][S][R];
		nzv= new int[K][I];
        nsi= new int[S];
		
		miu = new double[R][2];
		sigma = new double[R][2];
		
		miuT= new double[S];
		sigmaT= new double[S];

		
		phisum = new int [U][S]; //对应nusz;
		thetasum = new int [U][S]; //对应nusr;
		nzvsum= new int[K];
         
		test = Dataset.readSamples(input.testfile,input.geofile);
		testmap=Dataset.readTest(input.testfile);
	
	}
	
	public void initialize(){
		System.out.println("initializing model...");
		Random rand = new Random();
		for(int u=0;u<U;u++){
			if(trainmap.containsKey(u)){
				int[] Nu=train.trainI[u];
				Zu[u]=new int [Nu.length];
				Ru[u]=new int[Nu.length];
				Su[u]=new int[Nu.length];
				for(int i=0;i<Nu.length;i++){
					int z = rand.nextInt(K);
			    	int r = rand.nextInt(R);
			    	int s = rand.nextInt(S);
			    	Zu[u][i]=z;
			    	Su[u][i]=s;
			    	Ru[u][i]=r;
			    	Reg[r]++;
			    	nsi[s]++;
			    	nzv[z][Nu[i]]++;
			    	nusz[u][s][z]++;
			        nusr[u][s][r]++;
			    	phisum[u][s]++; 
			    	thetasum[u][s]++; 
			    	nzvsum[z]++;
			  }
			}			
		}		
		  
		for(int r=0;r<R;r++){
			miu[r][0] = 33.7+rand.nextFloat();
			miu[r][1] = -118.3+rand.nextFloat();
			sigma[r][0] = rand.nextFloat()*10;
			sigma[r][1] = rand.nextFloat()*10;
		}
		
		for(int s=0;s<S;s++){
			miuT[s]=rand.nextFloat();
			sigmaT[s]=rand.nextFloat();
		}
			 
	}
	
	
	 public void gibbs1(){
         for(int u=0;u<U;u++){
        	int[] Nu=train.trainI[u];
 			for(int i=0;i<Nu.length;i++){
 				double[] p = new double[K];
 				int z= Zu[u][i];
 				int s= Su[u][i];
 				double t= train.trainJimT[u][i];
 				nusz[u][s][z]--;
 				nzv[z][Nu[i]]--;
 				nzvsum[z]--;				
 				phisum[u][s]--;
 				for(int k=0;k<K;k++){
 					p[k]=(nusz[u][s][k]+beta)/(phisum[u][s]+K*beta)*(nzv[k][Nu[i]]+gamma)/(nzvsum[k]+K*gamma)
	 							*nsi[s]*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
 				}
 				Util.norm(p);
 				z = draw(p);
 				Zu[u][i]=z;
 				nusz[u][s][z]++;
 				nzv[z][Nu[i]]++;
 				nzvsum[z]++;
 				phisum[u][s]++;
 				
 			}
         }	 
	 }
	
	 public void gibbs2(){
		 for(int u=0;u<U;u++){
	       int[] Nu=train.trainI[u];
	 	   for(int i=0;i<Nu.length;i++){
	 		  double[] p = new double[R];
	 			 int eid=train.trainI[u][i];
	 		     int s= Su[u][i];
	 		     int r= Ru[u][i];
	 		     Reg[r]--;
	 			 double t= train.trainJimT[u][i];
	 			 nusr[u][s][r]--;
	 			 thetasum[u][s]--;
	 			 for(int k=0;k<R;k++){
	 				p[k]=(nusr[u][s][k]+alpha)/(thetasum[u][s]+R*alpha)
			 				  *pdf(eid,k)
			 				  *nsi[s]*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
	 			 }
	 			Util.norm(p);
 				r = draw(p);
 				Ru[u][i]=r;
 				nusr[u][s][r]++;
 				thetasum[u][s]++;	
 				Reg[r]++;
	 		 }
		 }
		 for(int r=0;r<R;r++)
			 updateRegGaussian(r);
	 }
	 
	 public void gibbs3(){
		 for(int u=0;u<U;u++){
		       int[] Nu=train.trainI[u];
		 	   for(int i=0;i<Nu.length;i++){		 		   
		 		  double[] p = new double[S]; 
		 		  int s= Su[u][i];
		 		  int r= Ru[u][i];
		 		  int z= Zu[u][i];
		 		  nusz[u][s][z]--;
		 		  phisum[u][s]--;
		 		  nusr[u][s][r]--;
	 			  thetasum[u][s]--;
	 			  nsi[s]--;
	 			  double t=train.trainJimT[u][i];
	 			  for(int k=0;k<S;k++){
	 				 p[k]=nsi[k]*1.0/Math.sqrt(2*Math.PI*sigmaT[k])*Math.exp(-(t-miuT[k])*(t-miuT[k])/(2*sigmaT[k]));
	 			  }
	 			 Util.norm(p);
	 			 s = draw(p);
	 			 Su[u][i]=s;
	 			 nusz[u][s][z]++;
		 		 phisum[u][s]++;
		 		 nusr[u][s][r]++;
	 			 thetasum[u][s]++;
	 			 nsi[s]++;
		 	   }
		   }
		 for(int s=0;s<S;s++)
			 updateTemGaussian(s);
	 }
	 public double[][] Parameter2(int[][] num,int[] sum,double sp){
			double[][] parameter = new double[num.length][num[0].length];
			for(int i = 0;i<num.length;i++){
				for(int j = 0;j<num[0].length;j++){
					parameter[i][j] = (num[i][j]+sp-1)/(sum[i]+num[i].length*sp-num[i].length);
				}
			}
			return parameter;	
		}
	 public double[][] estParameter(int[][] m, int[] sum, double sp){
			double[][] p = new double[m.length][m[0].length];		
			for(int i=0;i<m.length;i++)
				for(int j=0;j<m[i].length;j++){
					p[i][j] = (m[i][j]+sp)/(sum[i]+m[i].length*sp);
				}
					
			return p;
		}
	 
	 //u,s,z
	 public double[][][] estParameter1(int[][][] m, int[][] sum, double sp){
			double[][][] p = new double[U][][];
			for(int u=0;u<U;u++){
				p[u]=new double[m[u].length][m[u][0].length];
				for(int i=0;i<m[u].length;i++){
					for(int k=0;k<m[u][0].length;k++){
					  p[u][i][k]=(m[u][i][k]+sp)/(sum[u][i]+m[u][0].length*sp);
			
					}
				}
			}
			return p;
		}

		public int draw(double[] a){
			double r = Math.random();
			for(int i = 0; i<a.length;i++){
				r = r - a[i];
				if(r<0) return i;
			}
			return a.length-1;
		}
	
		public void updateRegGaussian(int r){
			 miu[r][0]=0;
			 miu[r][1]=0;
			 sigma[r][0]=0.01;
			 sigma[r][1]=0.01;
			 
			 List<Integer> lr = new ArrayList<Integer>();
				for(int u=0;u<train.trainI.length;u++){
					for(int or=0;or<train.trainI[u].length;or++){
						if(Ru[u][or]==r)
							lr.add(train.trainI[u][or]);
					}
				  }

				if(lr.size()<=1)
					return;
				for(int e : lr){
					miu[r][0]+= geo[e][0];
					miu[r][1]+= geo[e][1];
				}
				miu[r][0] /= lr.size();
				miu[r][1] /= lr.size();
				for(int e: lr){
					sigma[r][0] += (geo[e][0]-miu[r][0])*(geo[e][0]-miu[r][0]);
					sigma[r][1] += (geo[e][1]-miu[r][1])*(geo[e][1]-miu[r][1]);
				}
				sigma[r][0] /= lr.size();
				 
				sigma[r][1] /= lr.size();
			}
		 
		 
		 public void updateTemGaussian(int s){
			 
			 miuT[s]=0;
			 sigmaT[s]=0;
			for(int u=0;u<train.trainJimT.length;u++){
				for(int or=0;or<train.trainJimT[u].length;or++){
					if(Su[u][or]==s){
						miuT[s]+= train.trainJimT[u][or];
					}
					
				}
			}		 
			 miuT[s]/=nsi[s];
			 for(int u=0;u<train.trainJimT.length;u++){
					for(int or=0;or<train.trainJimT[u].length;or++){
						if(Su[u][or]==s){
							sigmaT[s]+= (train.trainJimT[u][or]-miuT[s])*(train.trainJimT[u][or]-miuT[s]);
						}
						
					}
				}	
			 sigmaT[s]/=nsi[s];

			}
		 public double pdf(int e, int r){
				double x = geo[e][0] - miu[r][0];
				double y = geo[e][1] - miu[r][1];
				double temp = Math.exp(-0.5*((x*x)/(sigma[r][0]*sigma[r][0])+(y*y)/(sigma[r][1]*sigma[r][1])));
				return temp/(2*3.142*Math.sqrt(sigma[r][0]*sigma[r][1]));
			}
		
		 
		 
		 public int[] returnReg(double[][]omiga){
			 int[] pReg=new int[I];
				 for(int p=0;p<I;p++){
					 double max=0;
					 int r=0;
					 for(int i=0;i<R;i++){
						if (max< omiga[i][p]){
							max=omiga[i][p];
							r=i;
						}
				 }
					 pReg[p]=r;
			 }			 
			 return pReg;
		 }
		
		public Model getModel() throws IOException{
			Model model = new Model();
			model.phiUS = estParameter1(nusz,phisum,beta);
			model.thetaUS = estParameter1(nusr,thetasum,alpha);
			model.etaZ=estParameter(nzv,nzvsum,gamma);
            model.miu=miu;
            model.sigma=sigma;
            model.miuT=miuT;           
            model.sigmaT=sigmaT;
          
            String filePath1="D:\\dataset\\USTTM\\LA\\parameters\\eta.txt";
    		output(model.etaZ,filePath1);

    		String filePath2="D:\\dataset\\USTTM\\LA\\parameters\\phi.txt";
    		output1(model.thetaUS,filePath2);

    		String filePath3="D:\\dataset\\USTTM\\LA\\parameters\\theta.txt";
    		output1(model.phiUS,filePath3);
			return model;
		}
		public void output(double[][] parameters,String filePath)throws IOException{
			 File write = new File(filePath);
		     BufferedWriter bw = new BufferedWriter(new FileWriter(write));
		     for(int i=0;i<parameters.length;i++){
		    	 for(int j=0;j<parameters[i].length;j++){
		    		bw.write(String.valueOf(parameters[i][j])+" " );
		    	 }
		    	 bw.write("\r");
		     }
		     bw.close();
		 }
		 
		 public void output1(double[][][] parameters,String filePath)throws IOException{
			 File write = new File(filePath);
		     BufferedWriter bw = new BufferedWriter(new FileWriter(write));
		     for(int i=0;i<parameters.length;i++){
		    	 for(int j=0;j<parameters[i].length;j++){
		    		 for(int k=0;k<parameters[i][j].length;k++){
		    			 bw.write(String.valueOf(parameters[i][j][k])+" " ); 
		    		 }
		    		
		    	 }
		    	 bw.write("\r");
		     }
		     bw.close();
		 }
		public Map<Integer,Map<Integer,int[]>> recommend(Model model,int topn){
			double[][][] score = new double[U][][];
			int[] pReg= returnReg(model.etaZ);
			for(int uid:testmap.keySet()){
				score[uid]= new double[test.trainJimT[uid].length][I];
				for(int b=0;b<testmap.get(uid).size();b++){
					double t=test.trainJimT[uid][b];
					int ur=pReg[testmap.get(uid).get(b)];
					for(int i=0;i<I;i++){
						if(trainmap.get(uid)==null||trainmap.get(uid).contains(i)) continue;
						int reg=pReg[i];
						if(ur==reg){
							double gsocre=0;
							double sscore=0;
							for(int s=0;s<S;s++){
								sscore=nsi[s]*1.0/Math.sqrt(2*Math.PI*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/(2*sigmaT[s]));
								double zscore=0;
								for(int k=0;k<K;k++){
									zscore+=(model.phiUS[uid][s][k]*model.etaZ[k][i]);
								}
								
								double rscore=0;
								for(int r=0;r<R;r++){
									rscore+=model.thetaUS[uid][s][r]*pdf(i,r);	
								}
								
								gsocre+=sscore*zscore*rscore;
							}
							
							score[uid][b][i]=gsocre;
							
						}
	
					}				
				}
				System.out.println("user"+uid+" "+"complete");
			}

			Map<Integer,Map<Integer,int[]>>list= new HashMap<Integer,Map<Integer,int[]>>();
			for(int uid=0;uid<test.trainJimT.length;uid++){
				for(int i=0;i<test.trainJimT[uid].length;i++){
					int[] index = Util.argSort(score[uid][i], topn);
					if(!list.containsKey(uid)){
						Map<Integer,int[]> tem= new HashMap<Integer,int[]>();
						tem.put(i, index);
						list.put(uid, tem);
					}else{
						Map<Integer,int[]>tem= list.get(uid);
						tem.put(i, index);
					}
				}
				
			}	

			return list;
		}
		
		public double evaluate(Map<Integer,Map<Integer,int[]>> lists,String metric){
			if("Recall".equals(metric)){
				double hit = 0;
				double recall = 0;
				for(int uid:lists.keySet()){
					for(int order:lists.get(uid).keySet()){
						recall+=1;
						int[] index=lists.get(uid).get(order); 
						int e=test.trainI[uid][order];
						for(int p:index){
							if(p==e){
								hit++;
							}
						}
					}
				}
				System.out.printf("recall=%s\n",recall);
                System.out.printf("hit=%s\n",hit);
				recall = hit/recall;
				System.out.printf("Recall=%f\n",recall);
				return recall;
			}else {
				double ndcg = 0;
				int count = 0;
				for(int uid:testmap.keySet()){
					List<Integer> slist=testmap.get(uid);
					for(int i=0;i<slist.size();i++){
						List<Integer> tem=new ArrayList<Integer>();
						tem.add(slist.get(i));
						double ndcg1=Util.ndcg(lists.get(uid).get(i), tem);
						
						if(ndcg1>0){
							ndcg += ndcg1;
							count++;
						}
					}
				}
				System.out.println("nDCG = "+ndcg/count);
				return ndcg/count;

			}
		}	
		public static void main(String[] args) throws IOException, ParseException {
			int K = 180; 
			int R=50;
			int S=5;
			String matrix = "LA";
			Input input = new Input(matrix);
			usttm2  com = new usttm2(input,K,R,S);
//			System.out.println(com.R);
			com.initialize();
			for(int iter=0;iter<com.iterNum;iter++){
				com.gibbs1();
				com.gibbs2();
				com.gibbs3();
				System.out.println("iteration "+iter);
			}
			Model model = com.getModel();
			Map<Integer,Map<Integer,int[]>> list = com.recommend(model,5);
            
			com.evaluate(list, "Recall");
			com.evaluate(list, "ndcg");
		}	
	
}
