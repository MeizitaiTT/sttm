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


import org.apache.commons.math3.distribution.NormalDistribution;

public class Usttm {
	int U;//�û���
	int K;//������
	int R;//������
	int S;//Ǳ��ʱ����
	
	int V;//poi��Ŀ
	//int Nu[];//ÿ���û���ǩ����Ŀ
	
	double[][] miuR;//���������ֵ
	double[][] sigmaR;//�������귽��
	double[] miuT;//ʱ��ľ�ֵ
	double[] sigmaT;//ʱ��ķ���
	
	double alpha;
	double beta;
	double gamma;
	
	//int[] u;
	//int[][] s;
	
	//double[][] cui1;
	//double[][] cui2;
	//double[][] tui;
	
	//int[] etasum;//����z��poi����
	
	int[][] Zu;//�û�u��i��ǩ����Ǳ������
	int[][] Ru;//�û�u��i��ǩ����Ǳ�ڵ���
	int[][] Su;//�û�u��i��ǩ����Ǳ��ʱ��
 	int[][][] musi;//�û�u�ڵ�s��ʱ����������z����
	int[][][] nusi;//�û�u�ڵ�s��ʱ�����ĵ���r����
	int[][] thetasum;//�û�u�ڵ�s��ʱ���������������
	int[][] phisum;//�û�u�ڵ�s��ʱ�������е�������
	int[] Ssum;
	int[][] Cz;//����z����poi��v������		
	int C[] ;//����z��poi����
	int nrv[][];//ÿ�������poi��v����Ŀ
	int nr[];//����r��poi����
	
	Train train;
	Train test;
	
	Map<Integer,List<Integer>> trainmap;//(ѵ����uid,ѵ�����û�uidȥ����poi�㼯��)
	Map<Integer,List<Integer>> testmap;//(���Լ�uid,���Լ��û�uidȥ����poi�㼯��)
	double[][] geo;
	
	public Usttm(int K,int R,int S,Input input) throws IOException, ParseException{
		this.K = K;
		this.R = R;
		this.S = S;
		
		alpha = 50.0/R;
		beta = 50.0/K;
		gamma = 0.1;
		
	    geo =Dataset.readGeo(input.geofile);
	    train = Dataset.readSamples(input.trainfile,input.geofile);
	    trainmap = Dataset.readTest(input.trainfile);
	    
		test = Dataset.readSamples(input.testfile,input.geofile);
		testmap=Dataset.readTest(input.testfile);
		
		//tui = train.trainJimT;
		//System.out.println(tui[0][0]);
		/*for(int i = 0;i<tui.length;i++){
			for(int j = 0; j<tui[i].length;j++){
				System.out.println(tui[i][j]);
			}
		}*/
		V = Dataset.V;
		U = Dataset.U;
		
		Zu = new int[U][];
		Ru = new int[U][];
		Su = new int[U][];
		
		musi = new int[U][S][K];
		nusi = new int[U][S][R];
		thetasum = new int[U][S];
		phisum = new int[U][S];

		miuR=new double [R][2];
		sigmaR=new double [R][2];

		Ssum = new int[S];
		
		
		//cui1 = new double[U][];
		//cui2 = new double[U][];
		//tui = new double[U][];
		
		sigmaT = new double[S];
		miuT = new double[S];
		Cz = new int[K][V];
		C = new int[K];
		
		nrv = new int[R][V];
		nr = new int[R];
	}
	public void initial(){
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
			    	//Reg[r]++;
			    	Ssum[s]++;
			    	Cz[z][Nu[i]]++;
			    	musi[u][s][z]++;
			        nusi[u][s][r]++;
			    	phisum[u][s]++; 
			    	thetasum[u][s]++; 
			    	C[z]++;
			    	nrv[r][Nu[i]]++;
			    	nr[r]++;
			  }
			}			
		}		
		  
		for(int r=0;r<R;r++){
			miuR[r][0] = 33.7+rand.nextFloat();
			miuR[r][1] = -118.3+rand.nextFloat();
			sigmaR[r][0] = rand.nextFloat()*10;
			sigmaR[r][1] = rand.nextFloat()*10;
		}
		
		for(int s=0;s<S;s++){
			miuT[s]=rand.nextFloat();
			sigmaT[s]=rand.nextFloat();
		}
			 
	}
	
	/*public void gibbs1(){
		for(int u = 0;u < U;u++){
			int[] Nu = train.trainI[u];
			for(int i = 0;i<Nu.length;i++){
				double[] p = new double[K];
				int z = Zu[u][i];
				int s = Su[u][i];
				double t = train.trainT[u][i];
				musi[u][s][z]--;
				thetasum[u][s]--;
				for(int k = 0;k<K;k++){
					//System.out.println(train.trainJimT[u][i]);
					p[k] = (musi[u][s][k]+beta)/(thetasum[u][s]+K*beta);
					
					p[k] = p[k]*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
					//System.out.println(p[k]);
					p[k] = p[k]*(Cz[k][Nu[i]]+gamma-1)/(C[k]+V*gamma-V);
					//*gaussianSampler(train.trainJimT[u][i], (double)s)*(Cz[z][Nu[i]]+gamma-1)/(C[z]+V*gamma-V);
				}
				Util.norm(p);
				Cz[z][Nu[i]]--;
				C[z]--;
				z = draw(p);			
				Zu[u][i] = z;				
				musi[u][s][z]++;
				thetasum[u][s]++;
				Cz[z][Nu[i]]++;
				C[z]++;
			}
						
		}
	}*/
	public void gibbs1(){
        for(int u=0;u<U;u++){
       	int[] Nu=train.trainI[u];
			for(int i=0;i<Nu.length;i++){
				double[] p = new double[K];
				int z= Zu[u][i];
				int s= Su[u][i];
				double t= train.trainT[u][i];
				musi[u][s][z]--;
				Cz[z][Nu[i]]--;
				C[z]--;				
				thetasum[u][s]--;
				for(int k=0;k<K;k++){
					p[k]=(musi[u][s][k]+beta)/(thetasum[u][s]+K*beta)*(Cz[k][Nu[i]]+gamma)/(C[k]+K*gamma)
	 							*Ssum[s]*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
				}
				Util.norm(p);
				z = draw(p);
				Zu[u][i]=z;
				musi[u][s][z]++;
				Cz[z][Nu[i]]++;
				C[z]++;
				thetasum[u][s]++;			
			}
        }	 
	 }
	public void updateTimeGaussian(int s){//ʱ���˹�ֲ��ľ�ֵ������
		miuT[s] = 0;
		sigmaT[s] = 0;
		for(int i = 0;i<train.trainT.length;i++){
			for(int u = 0;u<train.trainT[i].length;u++){
				if(Su[i][u] == s){
					miuT[s] += train.trainT[i][u];
				}
			}
		}
		miuT[s]/=Ssum[s];
		 for(int u=0;u<train.trainT.length;u++){
				for(int or=0;or<train.trainT[u].length;or++){
					if(Su[u][or]==s){
						sigmaT[s]+= (train.trainT[u][or]-miuT[s])*(train.trainT[u][or]-miuT[s]);
					}					
				}
			}	
		 sigmaT[s]/=Ssum[s];
		
	}
	public void updateRegGaussian(int r){//��ÿ�ε�����õ�����r�������е�poi����������ȡr����������ƽ��ֵ�뷽��
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
			for(int e : lr){//�����������������poi��������
				miuR[r][0]+= geo[e][0];
				miuR[r][1]+= geo[e][1];
			}
			miuR[r][0] /= lr.size();//�õ�ƽ��ֵ
			miuR[r][1] /= lr.size();
			for(int e: lr){
				sigmaR[r][0] += (geo[e][0]-miuR[r][0])*(geo[e][0]-miuR[r][0]);
				sigmaR[r][1] += (geo[e][1]-miuR[r][1])*(geo[e][1]-miuR[r][1]);
			}
			sigmaR[r][0] /= lr.size();
			sigmaR[r][1] /= lr.size();//�õ�����
		}
	/*public void gibbs2(){//Ϊÿ��ǩ���ó������ʵĵ���
		for(int u = 0;u < U;u++){
			int[] Nu = train.trainI[u];
			for(int i = 0; i<Nu.length; i++){
				double[] p = new double[R];
				int eid = train.trainI[u][i];
				int s = Su[u][i];
				int r = Ru[u][i];		
				double t = train.trainT[u][i];
				nusi[u][s][r]--;
				phisum[u][s]--;
				for(int k = 0;k<R;k++){
					//System.out.println(cui1[u][i]);
					//p[k] = pdf(eid, k);
					//p[k] = Math.exp(-1*(Math.pow(cui1[u][i]-miuR[r][0], 2)/(2*sigmaR[r][0]))-Math.pow(cui2[u][i]-miuR[r][1], 2)/(2*sigmaR[r][1]));
					//p[k] = p[k]/(2*Math.PI*Math.pow(sigmaR[r][0]*sigmaR[r][1], 1/2));
					//p[k] = p[k] * (nusi[u][s][k]+alpha)/(phisum[u][s]+R*alpha)*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
					p[k]=(nusi[u][s][k]+alpha)/(phisum[u][s]+R*alpha)
			 				  *pdf(eid,k)
			 				  *1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
				}
				Util.norm(p);
				r = draw(p);
				Ru[u][i] = r;			
				nusi[u][s][r]++;
				phisum[u][s]++;
			}
		}
		for(int r = 0;r<R;r++){
			updateRegGaussian(r);
		}
	}*/
	public void gibbs2(){
		 for(int u=0;u<U;u++){
	       int[] Nu=train.trainI[u];
	 	   for(int i=0;i<Nu.length;i++){
	 		  double[] p = new double[R];
	 			 int eid=train.trainI[u][i];
	 		     int s= Su[u][i];
	 		     int r= Ru[u][i];
	 		    // Reg[r]--;
	 			 double t= train.trainT[u][i];
	 			 nrv[r][Nu[i]]--;
	 			 nr[r]--;
	 			 nusi[u][s][r]--;
	 			 phisum[u][s]--;
	 			 for(int k=0;k<R;k++){
	 				p[k]=(nusi[u][s][k]+alpha)/(phisum[u][s]+R*alpha)
			 				  *pdf(eid,k)
			 				  *Ssum[s]*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s]);
	 			 }
	 			Util.norm(p);
				r = draw(p);
				nrv[r][Nu[i]]++;
	 			nr[r]++;
				Ru[u][i]=r;
				nusi[u][s][r]++;
				phisum[u][s]++;	
				//Reg[r]++;
	 		 }
		 }
		 for(int r=0;r<R;r++)
			 updateRegGaussian(r);
	 }
	public void gibbs3(){//Ϊÿ��ǩ��������������ʱ����
		for(int u=0;u<U;u++){
		       int[] Nu=train.trainI[u];
		 	   for(int i=0;i<Nu.length;i++){		 		   
		 		  double[] p = new double[S]; 
		 		  int s= Su[u][i];
		 		  int r= Ru[u][i];
		 		  int z= Zu[u][i];
		 		  musi[u][s][z]--;
		 		  phisum[u][s]--;
		 		  nusi[u][s][r]--;
	 			  thetasum[u][s]--;
	 			  Ssum[s]--;
	 			  double t=train.trainT[u][i];
	 			  for(int k=0;k<S;k++){
	 				 p[k]=Ssum[k]*1.0/Math.sqrt(2*Math.PI*sigmaT[k])*Math.exp(-(t-miuT[k])*(t-miuT[k])/(2*sigmaT[k]));
	 			  }
	 			 Util.norm(p);
	 			 s = draw(p);
	 			 Su[u][i]=s;
	 			 musi[u][s][z]++;
		 		 phisum[u][s]++;
		 		 nusi[u][s][r]++;
	 			 thetasum[u][s]++;
	 			 Ssum[s]++;
		 	   }
		   }
		 for(int s=0;s<S;s++)
			 updateTimeGaussian(s);
	}
	public int draw(double[] a){
			double r = Math.random();
			for(int i = 0; i<a.length;i++){
				r = r - a[i];
				if(r<0) return i;
			}
			return a.length-1;
	}
	 
	
	
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
	public Model getModel() throws IOException{
		Model model = new Model();
		model.eta = estParameter(Cz, C, gamma);//����k��poi��v�ĸ���
		String filePath1="D:\\dataset\\USTTM\\LA\\parameters\\eta.txt";
		//output(model.eta,filePath1);
		model.phi = estParameter1(nusi, phisum, alpha);//�û���ʱ����s�ڵ���r�ĸ���
		String filePath2="D:\\dataset\\USTTM\\LA\\parameters\\phi.txt";
		//output1(model.phi,filePath2);
		model.theta = estParameter1(musi, thetasum, beta);//�û���ʱ����s������r�ĸ���
		String filePath3="D:\\dataset\\USTTM\\LA\\parameters\\theta.txt";
		//output1(model.theta,filePath3);
		model.etaR = estParameter(nrv, nr, gamma);
		return model;
	}
	public int[] returnReg(double[][]omiga){
		 int[] pReg=new int[V];
			 for(int p=0;p<V;p++){
				 double max=0;
				 int r=0;
				 for(int i=0;i<R;i++){
					if (max< omiga[i][p]){
						max = omiga[i][p];
						r=i;
					}
			 }
				 pReg[p]=r;
		 }		 
		 return pReg;
	 } 
	public double pdf(int e, int r){//�����ĸ�˹�ֲ���ʽ
		double x = geo[e][0] - miuR[r][0];
		double y = geo[e][1] - miuR[r][1];
		double temp = Math.exp(-0.5*((x*x)/(sigmaR[r][0]*sigmaR[r][0])+(y*y)/(sigmaR[r][1]*sigmaR[r][1])));
		return temp/(2*3.142*Math.sqrt(sigmaR[r][0]*sigmaR[r][1]));
	}
	
	public Map<Integer,Map<Integer,int[]>> recommend(Model model,int topn){
		double[][][] score = new double[U][][];
		int[] pReg= returnReg(model.etaR);
		for(int uid:testmap.keySet()){//�������Լ�ÿ��ǩ��
			score[uid]= new double[test.trainT[uid].length][V];//ÿ���û�ÿ��ǩ����ÿ��POI�ĸ���
			for(int b=0;b<testmap.get(uid).size();b++){//����
				double t=test.trainT[uid][b];
				int ur=pReg[testmap.get(uid).get(b)];
				for(int i=0;i<V;i++){//����ÿ��poi��ĸ���
					if(trainmap.get(uid)==null||trainmap.get(uid).contains(i)) continue;//���ѵ�������޸��û�����ѵ�����и��û��Ѿ�ȥ����poi��(���Ѿ������˿�����)��������
					int reg=pReg[i];//��ȡ�����������ֵ
					if(ur==reg){//�˴���������
						double gsocre=0;
						double sscore=0;
						for(int s=0;s<S;s++){
							sscore=Ssum[s]*1.0/Math.sqrt(2*Math.PI*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/(2*sigmaT[s]));//��˹���ʼ���ʱ�����
							double zscore=0;
							for(int k=0;k<K;k++){
								zscore+=(model.theta[uid][s][k]*model.eta[k][i]);
							}						
							double rscore=0;
							for(int r=0;r<R;r++){
								rscore+=model.phi[uid][s][r]*pdf(i,r);	
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
		for(int uid=0;uid<test.trainT.length;uid++){
			for(int i=0;i<test.trainT[uid].length;i++){
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
	
	public static void main(String[] args) throws IOException, ParseException{
		int K = 80;
		int R = 200;
		int S = 10;
		String matrix = "LA";
		Input input = new Input(matrix);
		Usttm com = new Usttm(K,R,S,input);
		com.initial();
		for(int i = 0;i<100;i++){
			com.gibbs1();
			com.gibbs2();
			com.gibbs3();
			System.out.println("������"+i+"��");
		}
		Model model = com.getModel();
		Map<Integer,Map<Integer,int[]>> list = com.recommend(model,5);
        	for(Integer eid:list.keySet()){   		
        		Map<Integer,int[]> tem = list.get(eid);
        		for(Integer count:tem.keySet()){
        			int[] at = tem.get(count);
        			for(int k = 0;k<at.length;k++){
        				System.out.println("�û�"+eid+"��"+(k+1)+"�η����Ƽ�poi��idΪ"+at[k]);
        			}
        		}
        	}
		com.evaluate(list, "Recall");
		com.evaluate(list, "ndcg");
		
		/*for(int i = 0;i<model.eta.length;i++){
			for(int j = 0;j<model.eta[i].length;j++){
				System.out.println(model.eta[i][j]);
			}
		}*/
		/*for(int i = 0;i<model.phi.length;i++){
			for(int j = 0;j<model.phi[i].length;j++){
				for(int k = 0;k<model.phi[i][j].length;k++)
				System.out.println(model.phi[i][j][k]);
			}
		}*/
		//System.out.println(com.tui[1][0]);
	}
}
