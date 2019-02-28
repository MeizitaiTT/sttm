package JIM;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.distribution.BetaDistribution;

public class Jim {
	int K;
	int S ;//潜在时间间隔数
	int R;//潜在区域数量
	double alpha;
	double beta;
	double o=0.01;
	double varepsilon=0.01;
	
	int iterNum = 50;
	int U,I,W;
	int[][] Zu; //每个用户的主题
	int[][] Ru; //每个用户每次签到的区域
	int[] Reg;//每个区域的签到数量

	int [][] nzc; //每个主题z含有的主题个数

	int [][] nrv;//每个区域含有的POI个数
	
	
	
	double [][] miuR;
	double [][] sigmaR;
	double [][] Jimphi;

	
	int [][] nuz;
	int [][] nur;

	int[] phisum;
	int[] thetasum;

	int[] etasum;

	int[] omigasum;
    
	Train train;
	Train test;
	Map<Integer,List<Integer>> trainmap;
	Map<Integer,List<Integer>> testmap;
	//Map<Integer,Map<Double,Integer>> test;
	Map<Integer,String> poiCat;
	Map<Integer,List<Integer>> poiWd;
	double[][] geo; //每个位置的经纬度   
	Input input;
	public Jim(Input input,int K,int R) throws IOException{
		this.K = K;
		this.R = R;
		this.alpha=50.0/K;
		this.beta=50.0/R;
		
		this.input = input;
		poiCat= Dataset1.readPoiCat(input.geofile);
		poiWd=Dataset1.readPoiWord(input.geofile);
		geo = Dataset1.readGeo(input.geofile);
	    train = Dataset1.readSamples(input.trainfile,input.geofile);
	    trainmap = Dataset1.readTest(input.trainfile);
		U = Dataset1.U;
		I = Dataset1.I;
		W = Dataset1.W;
		System.out.println("单词数"+W);
		Zu=new int[U][] ; //每个用户的主题
		Ru=new int[U][]; //每个用户每次签到的区域

	    Reg=new int[R];//每个区域的签到数量

		nzc=new int [K][W]; //每个主题z含有的主题个数
		Jimphi=new double[K][2];
		
		nrv=new int [R][I];//每个区域含有的POI个数

		miuR=new double [R][2];
		sigmaR=new double [R][2];
		
		
		nuz=new int [U][K] ;
	    nur=new int [U][R];
		phisum=new int[U];
		thetasum=new int[U];

	    etasum=new int[K];
		omigasum=new int[R];
		
		
		//test=Dataset.readstamp(input.testfile,input.geofile);
		test = Dataset1.readSamples(input.testfile,input.geofile);
		testmap=Dataset1.readTest(input.testfile);
	}
	
	public void initialize(){
		System.out.println("initializing model...");
		Random rand = new Random();
		for(int u=0;u<U;u++){
			if(trainmap.containsKey(u)){
				int[] Nu=train.trainI[u];
				Zu[u]=new int [Nu.length];
				Ru[u]=new int[Nu.length];

				for(int i=0;i<Nu.length;i++){
					int z = rand.nextInt(K);
			    	int r = rand.nextInt(R);
			    	Zu[u][i]=z;
	
			    	Ru[u][i]=r;
			    	Reg[r]++;

			
			    	String[] cset=train.trainC[u][i].split("\\|");
			    	String cid=cset[1];
			    	String[] cidset=cid.split("\\.");
			    	for(String cd:cidset){
			    	   nzc[z][Integer.parseInt(cd)]++;
			    	   etasum[z]++;
			    	}
			    	

                    nrv[r][Nu[i]]++;
				    nuz[u][z]++;
				    phisum[u]++;
				    thetasum[u]++;
				    nur[u][r]++;
				    omigasum[r]++;

			  }
			}			
		}
		
		for(int r=0;r<R;r++){
			miuR[r][0] = 33.7+rand.nextFloat();
			miuR[r][1] = -118.3+rand.nextFloat();
			sigmaR[r][0] = rand.nextFloat()*10;
			sigmaR[r][1] = rand.nextFloat()*10;
		}
		 
		for(int k=0;k<K;k++){
			Jimphi[k][0]=rand.nextFloat();
			Jimphi[k][1]=rand.nextFloat();
		}
	}	
	
	 public void gibbs1(){
         for(int u=0;u<U;u++){
        	int[] Nu=train.trainI[u];
 			for(int i=0;i<Nu.length;i++){
 				double[] p = new double[K];
 				int z= Zu[u][i];
 				double t=train.trainJimT[u][i];
                String word=train.trainW[u][i];
                String[] wset=word.split("\\.");
                for(int w=0;w<wset.length;w++){
                	int wid=Integer.parseInt(wset[w]);
                	nzc[z][wid]--;
                	etasum[z]--;
                }
 				nuz[u][z]--;
 				
 				phisum[u]--;
 				for(int k=0;k<K;k++){
 					p[k]=(nuz[u][k]+alpha)/(phisum[u]+K*alpha)*Math.pow(1-t, Jimphi[k][0]-1)
 							*Math.pow(1-t, Jimphi[k][1]-1)/betasampler(Jimphi[k][0],Jimphi[k][1]);
 				
 					for(int w=0;w<wset.length;w++){
 	                	int wid=Integer.parseInt(wset[w]);
 	                	p[k]*=(nzc[k][wid]+varepsilon)/(etasum[k]+W*varepsilon);
 	                	
 	                }
 				}
 				Util.norm(p);
 				z = draw(p);
 				Zu[u][i]=z;

 				for(int w=0;w<wset.length;w++){
                	int wid=Integer.parseInt(wset[w]);
                	nzc[z][wid]++;
                	etasum[z]++;
                }
 				phisum[u]++;
 				nuz[u][z]++;
 			}
         }	 
         
         for (int k=0;k<K;k++)
         updateTimeBeta(k);
         
	 }
	 
	 public void updateTimeBeta(int z){
		 Jimphi[z][0]=0;
		 Jimphi[z][1]=0;
		 double miuZ=0;
		 double sigmaZ=0;
		 int count=0;
		 for(int u=0;u<train.trainJimT.length;u++){
				for(int or=0;or<train.trainJimT[u].length;or++){
					if(Zu[u][or]==z){
						miuZ+= train.trainT[u][or];
						count++;
					}
					
				}
			}	
		   if(count==0){
			   miuZ=0;
		   }else{
			   miuZ/=count;  
		   }
		   
			 for(int u=0;u<train.trainJimT.length;u++){
					for(int or=0;or<train.trainJimT[u].length;or++){
						if(Zu[u][or]==z){
							sigmaZ+= (train.trainJimT[u][or]-miuZ)*(train.trainJimT[u][or]-miuZ);
						}
						
					}
				}
			 if(count==0){
				 sigmaZ=1;
			   }else{
				   sigmaZ/=count;  
			   }
			 
			 Jimphi[z][0]=miuZ*((miuZ*(1-miuZ)/sigmaZ)-1);
			 
			 Jimphi[z][1]=(1-miuZ)*((miuZ*(1-miuZ)/sigmaZ)-1);
	 }
	 public static double betasampler(double alpha,double beta){
	        BetaDistribution Beta=new BetaDistribution(alpha,beta);
	        return Beta.sample();
	    }
	 public void gibbs2(){
		 
		 for(int u=0;u<U;u++){
	       int[] Nu=train.trainI[u];
	 	   for(int i=0;i<Nu.length;i++){
	 		     double[] p = new double[R];
	 			 int eid=Nu[i];
	 			 int r= Ru[u][i];

	 			 Reg[r]--;
	 			 
	 			 nrv[r][Nu[i]]--;
	 			
	 			 nur[u][r]--;
	 			 thetasum[u]--;
	 			 omigasum[r]--;
	 			 for(int k=0;k<R;k++){
	 				p[k]=(nur[u][k]+beta)/(thetasum[u]+R*beta)*(nrv[k][Nu[i]]+o)/(omigasum[k]+I*o)*pdf(eid,r);			 			
	 				
	 			 }
	 			Util.norm(p);
 				r = draw(p);
 				Ru[u][i]=r;
 				Reg[r]++;

	 			nrv[r][Nu[i]]++;
	 			nur[u][r]++;
	 			omigasum[r]++;
	 			thetasum[u]++;
	 		 }
		 }
		 
		 for(int r=0;r<R;r++)
			 updateRegGaussian(r);
	 }
	 
	 public void updateRegGaussian(int r){
		 miuR[r][0]=0;
		 miuR[r][1]=0;
		 sigmaR[r][0]=0.01;
		 sigmaR[r][1]=0.01;
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
				miuR[r][0]+= geo[e][0];
				miuR[r][1]+= geo[e][1];
			}
			miuR[r][0] /= lr.size();
			miuR[r][1] /= lr.size();
			for(int e: lr){
				sigmaR[r][0] += (geo[e][0]-miuR[r][0])*(geo[e][0]-miuR[r][0]);
				sigmaR[r][1] += (geo[e][1]-miuR[r][1])*(geo[e][1]-miuR[r][1]);
			}
			sigmaR[r][0] /= lr.size();
			sigmaR[r][1] /= lr.size();
		}
	 

	 public int draw(double[] a){
			double r = Math.random();
			for(int i = 0; i<a.length;i++){
				r = r - a[i];
				if(r<0) return i;
			}
			return a.length-1;
		}
	 
	 public double pdf(int e, int r){
			double x = geo[e][0] - miuR[r][0];
			double y = geo[e][1] - miuR[r][1];
			double temp = Math.exp(-0.5*((x*x)/(sigmaR[r][0]*sigmaR[r][0])+(y*y)/(sigmaR[r][1]*sigmaR[r][1])));
			return temp/(2*3.142*Math.sqrt(sigmaR[r][0]*sigmaR[r][1]));
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
	 
	 
	 
	 public double[][] estParameter(int[][] m, int[] sum, double sp){
			double[][] p = new double[m.length][m[0].length];
			
			for(int i=0;i<m.length;i++)
				for(int j=0;j<m[i].length;j++){
					p[i][j] = (m[i][j]+sp)/(sum[i]+m[i].length*sp);
				}
					
			return p;
		}
	 
	 public double[][][] estParameter1(int[][][] m, int[][] sum, double sp){
			double[][][] p = new double[m.length][][];
			
			for(int u=0;u<m.length;u++){	
				for(int i=0;i<m[u].length;i++){
					for(int k=0;k<m[i].length;k++){
					  p[u][i][k]=(m[u][i][k]+sp)/(sum[u][i]+m[i].length*sp);
			
					}
				}
			}
			return p;
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
	 
	 public void output2(double[][]parameters,String filePath)throws IOException{
		 File write = new File(filePath);
	     BufferedWriter bw = new BufferedWriter(new FileWriter(write));
	     for(int i=0;i<parameters.length;i++){
	    	 bw.write(String.valueOf(parameters[i][0])+" "+String.valueOf(parameters[i][1])+"\r");
	     }
	     bw.close();
	 }
	 
	 public void output3(double[]parameters,String filePath)throws IOException{
		 File write = new File(filePath);
	     BufferedWriter bw = new BufferedWriter(new FileWriter(write));
	     for(int i=0;i<parameters.length;i++){
	    	 bw.write(String.valueOf(parameters[i])+"\r");
	     }
	     bw.close();
	 }
	 
	 
	 public Model getModel() throws IOException{
		Model model = new Model();
		
		

		model.eta=estParameter(nzc,etasum,varepsilon);
		String filePath1="F:\\dataset\\TSSTM\\LA\\parameters\\eta.txt";
		//output(model.eta,filePath1);

		
		model.omiga=estParameter(nrv,omigasum,o);
		String filePath3="F:\\dataset\\TSSTM\\LA\\parameters\\omiga.txt";
		//output(model.omiga,filePath3);
		
		model.phi=estParameter(nuz,phisum,alpha);
		String filePath4="F:\\dataset\\TSSTM\\LA\\parameters\\phi.txt";
		//output(model.phi,filePath4);
		
		model.theta=estParameter(nur,thetasum,beta);
		String filePath5="F:\\dataset\\TSSTM\\LA\\parameters\\theta.txt";
		//output(model.theta,filePath5);
		
		
        model.miuR=miuR;
        String filePath8="F:\\dataset\\TSSTM\\LA\\parameters\\miuR.txt";
        //output2(model.miuR,filePath8);
        
        model.sigmaR=sigmaR;
        String filePath9="F:\\dataset\\TSSTM\\LA\\parameters\\sigmaR.txt";
        //output2(model.sigmaR,filePath9);
        
        model.Jimphi=Jimphi;
//        for(int k=0;k<K;k++){
//        	System.out.println(model.Jimphi[k][0]+"  "+ model.Jimphi[k][1]);
//        }
            
		return model;
		}
	 
	 public Map<Integer,Map<Integer,int[]>>  recommend(Model model,int topn) throws IOException{
			double[][][] score = new double[U][][];
			int[] pReg= returnReg(model.omiga);
			for(int i=0;i<testmap.size();i++){
				score[i]= new double[test.trainT[i].length][I];
                for(int m=0;m<testmap.get(i).size();m++){
                	int ur=pReg[testmap.get(i).get(m)];
                    double t=test.trainJimT[i][m];
    				for(int p=0;p<I;p++){
    					if(trainmap.get(i)==null||trainmap.get(i).contains(p)) continue;
    					int reg=pReg[p];
    					if(ur==reg){
    						List<Integer>wset=poiWd.get(p);
        					double sscore=0;
        					for(int k=0;k<K;k++){     						
        						double wscore=1;
        						for(int w=0;w<wset.size();w++){
        							wscore*=model.eta[k][w];
        						}
        						wscore=Math.pow(wscore, 1.0/wset.size());
        						wscore*=model.phi[i][k]*Math.pow(1-t, Jimphi[k][0]-1)
        	 							*Math.pow(1-t, Jimphi[k][1]-1)/betasampler(Jimphi[k][0],Jimphi[k][1]); 
        						sscore+=wscore;
        					}
        					double rscore = 0;
    						for(int r=0;r<R;r++){
    							rscore+=model.omiga[r][p]*pdf(p,r);
    						}     									
        					score[i][m][p]=sscore*rscore;     					
        				}
    				}
                }
				
				System.out.println("user"+i+" "+"complete");
			}


			Map<Integer,Map<Integer,int[]>>list= new HashMap<Integer,Map<Integer,int[]>>();
			for(int i=0;i<test.trainT.length;i++){
				for(int j=0;j<test.trainT[i].length;j++){
					int[] index=Util.argSort(score[i][j], topn);
					if(!list.containsKey(i)){
						Map<Integer,int[]> tem= new HashMap<Integer,int[]>();
						tem.put(j, index);
						list.put(i, tem);
					}else{
						Map<Integer,int[]>tem= list.get(i);
						tem.put(j, index);
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
	 
	 public static void main(String[] args) throws IOException {
			int K = 180; 
			int R = 50;
			String matrix = "LA";
			Input input = new Input(matrix);
			Jim com = new Jim(input,K,R);
			com.initialize();
			for(int iter=0;iter<com.iterNum;iter++){
				com.gibbs1();
				com.gibbs2();
				System.out.println("iteration "+iter);
			}
			Model model = com.getModel();
			Map<Integer,Map<Integer,int[]>> list = com.recommend(model,5);         
			com.evaluate(list, "Recall");
			com.evaluate(list, "ndcg");
		}
}
