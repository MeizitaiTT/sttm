package msttm;

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

import JIM.Dataset1;
import msttm.Dataset;
import msttm.Train;
import msttm.Util;



/**
 * @author 
 *
 */
public class Msttm {
	int U;// �û���
	int K;// ������
	int R;// ������
	int G;// ������
	int S;// Ǳ��ʱ����
	int V;// poi��
	int Q;// ������

	int[][] Zu;// �û�u��i��ǩ����Ǳ������
	int[][] Ru;// �û�u��i��ǩ����Ǳ�ڵ���
	int[][] Su;// �û�u��i��ǩ����Ǳ��ʱ��
	int[] Ssum;//����ʱ������POI����
	int[][][] musi;// �û�u�ڵ�s��ʱ����������z����
	int[][][] nusi;// �û�u�ڵ�s��ʱ�����ĵ���r����
	int[][] thetasum;// �û�u�ڵ�s��ʱ���������������
	int[][] phisum;// �û�u�ڵ�s��ʱ�������е�������
	int[][] Czq;// ����z�ĵ���q�ĸ���
	int[] Cz;// ����z�����е�����
	int[][] Crq;// ����r�ĵ���q�ĸ���
	int[] Cr;// ����r�����е�����
	
	int[][] mgz;// ����z�ڳ���g�Ĺ۲����
	int[] mg;// ����g�۲������������
	int[][] ngr;// ����r�ڳ���g�Ĺ۲����
	int[] ng;// ����g�۲�����е�����
	
	int[][] Czv;// ����z��poi��v������
	int C[];// ����z��poi����
	//int[] etasum;// ����z��poi����
	int nrv[][];//ÿ�������poi��v����Ŀ
	int nr[];//����r��poi����
	int[] Cpid;//ÿ��pid���ڳ���
	
	double[][] cui1;
	double[][] cui2;
	double[][] tui;

	double[][] miuR;// ���������ֵ
	double[][] sigmaR;// �������귽��
	
	double[] miuT;//ʱ��ľ�ֵ
	double[] sigmaT;//ʱ��ķ���
	double alpha;
	double beta;
	double gamma;
	double Lambda;
	double delta;
	double tau;
	double epsilon;
	Train train;
	Train test;

	Map<Integer, List<Integer>> trainmap;// (ѵ����uid,ѵ�����û�uidȥ����poi�㼯��)
	Map<Integer, List<Integer>> testmap;// (���Լ�uid,���Լ��û�uidȥ����poi�㼯��)
	double[][] geo;
	Map<Integer,List<Integer>> poiWd;
	public Msttm(int K, int R, int S, Input input) throws IOException, ParseException {
		this.K = K;
		this.R = R;
		this.S = S;

		alpha = 50.0 / R;
		beta = 50.0 / K;
		gamma = Lambda = delta = tau = epsilon = 0.1;

		geo = Dataset.readGeo(input.geofile);
		train = Dataset.readSamples(input.trainfile, input.geofile);
		trainmap = Dataset.readTest(input.trainfile);

		test = Dataset.readSamples(input.testfile, input.geofile);
		testmap = Dataset.readTest(input.testfile);
		Cpid = Dataset.getCpid(input.geofile);
		tui = train.trainJimT;
		
		poiWd = Dataset1.readPoiWord(input.geofile);
		
		this.U = Dataset.U;
		this.V = Dataset.V;
		this.Q = Dataset.Q;
		this.G = Dataset.G;
		Zu = new int[U][];
		Ru = new int[U][];
		musi = new int[U][S][K];
		nusi = new int[U][S][R];
		thetasum = new int[U][S];
		phisum = new int[U][S];
		//etasum = new int[K];
		miuR = new double[R][2];
		sigmaR = new double[R][2];
		Su = new int[U][];
		Ssum = new int[S];
		Czq = new int[K][Q];
		Cz = new int[K];
		Crq = new int[R][Q];
		Cr = new int[R];
		mgz = new int[G][K];
		mg = new int[G];
		ngr = new int[G][R];
		ng = new int[G];
		
		Czv = new int[K][V];
		C = new int[K];
		nrv = new int[R][V];
		nr = new int[R];
				

		
		sigmaT = new double[S];
		miuT = new double[S];
		
	}

	public void initial() {
		System.out.println("initializing model...");
		System.out.println("the number of city is "+G);
		System.out.println("the number of user is "+U);
		System.out.println("the number of POI is "+V);
		System.out.println("the number of word is "+Q);
		Random random = new Random();
		for (int u = 0; u < U; u++) {
			if (trainmap.containsKey(u)) {
				int[] Nu = train.trainI[u];// �û�Uǩ����ȫ��POI
				Zu[u] = new int[Nu.length];
				Ru[u] = new int[Nu.length];
				Su[u] = new int[Nu.length];
				for(int i = 0;i<Nu.length;i++){
					int city = Cpid[Nu[i]];
					int z = random.nextInt(K);
					int r = random.nextInt(R);
					int s = random.nextInt(S);				
					String[] cidset=train.trainC[u][i].split("\\.");//wordset
					//System.out.println("������Ϊ"+cidset.length);
			    	/*String cid=cset[1];//wordset
			    	String[] cidset=cid.split("\\.");*/
			    	for(String cd:cidset){
			    	   Czq[z][Integer.parseInt(cd)]++;//����z��ĵ���cd����Ŀ++
			    	   Cz[z]++;//����z�ĵ�����++
			    	   Crq[r][Integer.parseInt(cd)]++;//����r��ĵ���cd����Ŀ++
			    	   Cr[r]++;
			    	}					
					Zu[u][i] = z;
					Ru[u][i] = r;
					Su[u][i] = s;
					musi[u][s][z]++;
					nusi[u][s][r]++;;
					
					thetasum[u][s]++;
					phisum[u][s]++;
					
					Czv[z][Nu[i]]++;
					C[z]++;		
					
					nrv[r][Nu[i]]++;
					nr[r]++;
					
					mgz[city][z]++;
					mg[city]++;
					
					ngr[city][r]++;
					ng[city]++;
					
					Ssum[s]++;
				}
			}
		}
		for(int r=0;r<R;r++){
			miuR[r][0] = 33.7+random.nextFloat();
			miuR[r][1] = -118.3+random.nextFloat();
			sigmaR[r][0] = random.nextFloat()*10;
			sigmaR[r][1] = random.nextFloat()*10;
		}
		for(int s = 0;s<S;s++){
			sigmaT[s] = random.nextFloat();
			miuT[s] = random.nextFloat();
		}
	}
	
	public void gibbs1(){
		for(int u = 0 ; u<U ; u++){
			int[] Nu = train.trainI[u];
			for(int i = 0;i<Nu.length;i++){
				double[] p = new double[K];
				int z = Zu[u][i];
				int s = Su[u][i];
				int city = Cpid[Nu[i]];
				String word=train.trainW[u][i];
	            String[] wset=word.split("\\.");
	            double t = train.trainJimT[u][i];
				musi[u][s][z]--;
				thetasum[u][s]--;
				mgz[city][z]--;
				Czv[z][Nu[i]]--;
				Cz[z]--;
				for(int w = 0;w<wset.length;w++){
					int wid = Integer.parseInt(wset[w]);
					Czq[z][wid]--;
					Cz[z]--;
				}	
				for(int k = 0;k<K;k++){
					p[k] = (musi[u][s][k]+beta)/(thetasum[u][s]+beta*K);
					p[k] = p[k]*(mgz[city][k]+tau)/(mg[city]+tau*K)*1.0/Math.sqrt(2*3.1415*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s])*Ssum[s];
					p[k] = p[k]*(Czv[k][Nu[i]]+gamma)/(Cz[k]+gamma*V);
					for(int w = 0;w<wset.length;w++){
						int wid = Integer.parseInt(wset[w]);
						p[k] *= (Czq[k][wid]+epsilon)/(Cz[k]+epsilon*Q);
					}				
				}	
				Util.norm(p);							
				z = draw(p);				
				Zu[u][i] = z;				
				musi[u][s][z]++;
				thetasum[u][s]++;
				mgz[city][z]++;
				Czv[z][Nu[i]]++;
				Cz[z]++;
				for(int w = 0;w<wset.length;w++){
					int wid = Integer.parseInt(wset[w]);
					Czq[z][wid]++;
					Cz[z]++;
				}
			}
			
		}
	}
	public void gibbs2(){
		for(int u = 0 ; u<U ; u++){
			int[] Nu = train.trainI[u];
			for(int i = 0;i<Nu.length;i++){
				double[] p = new double[R];
				int eid=train.trainI[u][i];
				int r = Ru[u][i];
				int s = Su[u][i];
				int city = Cpid[Nu[i]];
				double t = train.trainJimT[u][i];
				String word=train.trainW[u][i];
	            String[] wset=word.split("\\.");
				nusi[u][s][r]--;
				phisum[u][s]--;
				ngr[city][r]--;	
				nrv[r][Nu[i]]--;
				nr[r]--;
				for(int w = 0;w<wset.length;w++){
					int wid = Integer.parseInt(wset[w]);
					Crq[r][wid]--;
					Cr[r]--;
				}
				for(int k = 0;k<R;k++){
					p[k] = (nusi[u][s][k]+alpha)/(phisum[u][s]+alpha*R);
					p[k] = p[k]*(ngr[city][k]+Lambda)/(ng[city]+Lambda*R)*Math.exp(-(t-miuT[s])*(t-miuT[s])/2*sigmaT[s])*Ssum[s];					
					for(int w = 0;w<wset.length;w++){
						int wid = Integer.parseInt(wset[w]);
						p[k] *= (Crq[k][wid]+delta)/(Cr[k]+delta*Q);
					}
					//p[k] *= Math.exp(-1*(Math.pow(cui1[u][i]-miuR[r][0], 2)/(2*sigmaR[r][0]))-Math.pow(cui2[u][i]-miuR[r][1], 2)/(2*sigmaR[r][1]));
					p[k]*=pdf(eid, k);
				}					
				Util.norm(p);							
				r = draw(p);
				//System.out.println(r);
				Ru[u][i] = r;				
				nusi[u][s][r]++;
				phisum[u][s]++;
				ngr[city][r]++;
				nrv[r][Nu[i]]++;
				nr[r]++;
				for(int w = 0;w<wset.length;w++){
					int wid = Integer.parseInt(wset[w]);
					Crq[r][wid]++;
					Cr[r]++;
				}
			}			
		}
		for(int r = 0;r<R;r++){
			updateRegGaussian(r);
		}
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
	public void updateTimeGaussian(int s){//ʱ���˹�ֲ��ľ�ֵ������
		miuT[s] = 0;
		sigmaT[s] = 0;
		for(int i = 0;i<train.trainJimT.length;i++){
			for(int u = 0;u<train.trainJimT[i].length;u++){
				if(Su[i][u] == s){
					miuT[s] += train.trainT[i][u];
				}
			}
		}
		miuT[s]/=Ssum[s];
		 for(int u=0;u<train.trainJimT.length;u++){
				for(int or=0;or<train.trainJimT[u].length;or++){
					if(Su[u][or]==s){
						sigmaT[s]+= (train.trainJimT[u][or]-miuT[s])*(train.trainJimT[u][or]-miuT[s]);
					}					
				}
			}	
		 sigmaT[s]/=Ssum[s];
		
	}
	public int draw(double[] a){
		double r = Math.random();
		for(int i = 0; i<a.length;i++){
			r = r - a[i];
			if(r<0) return i;
		}
		return a.length-1;
	}
	public Model getModel() throws IOException{
		Model model = new Model();
		model.eta = estParameter(Czv, C, gamma);//����k��poi��v�ĸ���
		String filePath1="D:\\dataset\\MSTTM\\LA\\parameters\\eta.txt";
		//output(model.eta,filePath1);
		model.phi = Parameter1(nusi, phisum, alpha);//�û���ʱ����s�ڵ���r�ĸ���
		String filePath2="D:\\dataset\\MSTTM\\LA\\parameters\\phi.txt";
		//output1(model.phi,filePath2);
		model.theta = Parameter1(musi, thetasum, beta);//�û���ʱ����s������r�ĸ���
		String filePath3="D:\\dataset\\MSTTM\\LA\\parameters\\theta.txt";
		
		model.chi = estParameter(mgz, mg, tau);
		model.omiga = estParameter(ngr, ng, Lambda);
		model.paiz = estParameter(Czq, Cz, epsilon);
		model.pair = estParameter(Crq, Cr, delta);
		model.etaZ = estParameter(nrv, nr, 0.01);
		String filePath4="D:\\dataset\\MSTTM\\LA\\parameters\\etaZ.txt";
		output(model.etaZ,filePath4);
		return model;
	}
	public double[][][] Parameter1(int[][][] num,int[][] sum,double sp){
		double[][][] p = new double[num.length][num[0].length][num[0][0].length];
		for(int i = 0;i<num.length;i++){
			for(int j = 0;j<num[i].length;j++){
				for(int k = 0;k<num[i][j].length;k++){
					p[i][j][k] = (num[i][j][k]+sp)/(sum[i][j]+num[i][j].length*sp);
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
		for(int uid:testmap.keySet()){//�������Լ�ÿ��ǩ��
			score[uid]= new double[test.trainT[uid].length][V];//ÿ���û�ÿ��ǩ����ÿ��POI�ĸ���
			for(int b=0;b<testmap.get(uid).size();b++){//����
				double t=test.trainJimT[uid][b];
				int ur=pReg[testmap.get(uid).get(b)];
				for(int i=0;i<V;i++){//����ÿ��poi��ĸ���
					if(trainmap.get(uid)==null||trainmap.get(uid).contains(i)) continue;//���ѵ�������޸��û�����ѵ�����и��û��Ѿ�ȥ����poi��(���Ѿ������˿�����)��������
					int reg=pReg[i];//��ȡ�����������ֵ
					int city = Cpid[i];
					if(ur==reg){//������poi�����poi��i��ͬһ����
						List<Integer>wset=poiWd.get(i);//��ȡ��poi�㣨����������Ե�poi�㣩�ĵ���
						double gsocre=0;
						double sscore=0;
						for(int s=0;s<S;s++){
							sscore=Ssum[s]*1.0/Math.sqrt(2*Math.PI*sigmaT[s])*Math.exp(-(t-miuT[s])*(t-miuT[s])/(2*sigmaT[s]));//��˹���ʼ���ʱ�����
							double zscore=0;						
							for(int k=0;k<K;k++){
								zscore+=(model.theta[uid][s][k]*model.eta[k][i]*model.chi[city][k]);
								double wscore=1;
	    						for(int w=0;w<wset.size();w++){//����Poi���е�������
	    							wscore*=model.paiz[k][w];//�õ���poi������ĵ����ڸ���������ʵĳ˻�
	    						}
	    						wscore=Math.pow(wscore, 1.0/wset.size());
	    						zscore*=wscore;
							}						
							double rscore=0;
							for(int r=0;r<R;r++){
								rscore+=model.phi[uid][s][r]*pdf(i,r)*model.omiga[city][r];	
								double wscore=1;
	    						for(int w=0;w<wset.size();w++){//����Poi���е�������
	    							wscore*=model.pair[r][w];//�õ���poi������ĵ����ڸ����������ʵĳ˻�
	    						}
	    						wscore=Math.pow(wscore, 1.0/wset.size());
	    						rscore*=wscore;
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
		int K = 250;
		int R = 250;
		int S = 10;
		String matrix = "LA";
		Input input = new Input(matrix);
		Msttm com = new Msttm(K,R,S,input);
		com.initial();
		for(int i = 0;i<200;i++) {
			com.gibbs1();
			com.gibbs2();
			com.gibbs3();
			System.out.println("������"+(i+1)+"��");
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
