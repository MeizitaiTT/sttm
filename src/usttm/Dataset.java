package usttm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Date;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Dataset {
	public static int U;//�û���Ŀ
	public static int V;//POI����
	
	/**
	 * @param fileName trainData
	 * @param fileName1 Datasetseq
	 * @return Train����������������ξ��󣬱�ʾ��������
	 * @throws IOException
	 * @throws ParseException 
	 */
	public static Train  readSamples(String fileName,String fileName1) throws IOException, ParseException{
		    Map<Integer,Double>stampsum=readstampsum(fileName1);//(uid,stamp(�����uid��stamp���))
		    Map<Integer,Integer> usercheck=readcheckinfo(fileName);//(uid,��uid��ǩ������)
            Map<Integer,Integer> usersq=new HashMap<Integer,Integer>();//(uid,uid��train�г��ֵĴ���-1)
            int [][]sample= new int[usercheck.size()][];  //�û�-����-��Ŀ
		    double[][] sample1=new double[usercheck.size()][];//�û�-����-��ǰʱ��������е�ʱ���ֵ
		    String[][] sample2=new String[usercheck.size()][];// �û�-����-���ʼ���
		    String[][] sample3=new String[usercheck.size()][];//�û�-����-���ϼ���
		    double[][] sample4=new double[usercheck.size()][];//�û�-����-��ǰʱ��
		    Train t= new Train();
		    File read = new File(fileName);
		    BufferedReader br = new BufferedReader(new FileReader(read));
	        String temp=br.readLine();
	        while(temp!=null){
	        	String[] str=temp.split("\t");
	        	int uid=Integer.parseInt(str[0]);
	        	String wd=str[3];
	        	//String cd=str[4];
	        	int num=usercheck.get(uid);//�õ�ǩ������
	        	if(!usersq.containsKey(uid)){//ͨ��ǩ��������ʼ�����ξ��󣬼���uid���û���numǩ�������ļ�¼
	        		sample[uid]=new int[num];	
		        	sample1[uid]=new double[num];
		        	sample2[uid]=new String[num];
		        	sample3[uid]=new String[num];
		        	sample4[uid]=new double[num];
		        	usersq.put(uid, 0);     	
	        	}else{
	        		usersq.put(uid, usersq.get(uid)+1);
	        	}
	        	
	        	Integer pid=Integer.parseInt(str[1]);
	        	sample[uid][usersq.get(uid)]=pid;//��uid���û��ĵ�usersq.get(uid)��ǩ����pid
				SimpleDateFormat sdf2 = new SimpleDateFormat("yyyy-MM-dd");
		    	Double tid=(double)sdf2.parse(str[2]).getTime();
	        	sample1[uid][usersq.get(uid)]=tid*1.0/stampsum.get(uid);//��uid�û��ĵ�usersq.get(uid)��ǩ����stampʱ���������stampʱ��֮��
	        	sample2[uid][usersq.get(uid)]=wd;//��uid�û��ĵ�usersq.get(uid)��ǩ���ĵ��ʼ���
	        	sample3[uid][usersq.get(uid)]=wd;//��uid�û��ĵ�usersq.get(uid)��ǩ���ļ��ϼ���
	        	
	        /*	long t1=Long.parseLong(str[2]);
	        	SimpleDateFormat sdf=new SimpleDateFormat("yyyy-MM-dd");
	        	String sd = sdf.format(new Date(t1));*/
	        	double month=Double.parseDouble(str[2].split("-")[2]);
	        	if(month==0){
	        		sample4[uid][usersq.get(uid)]=0.01;//��ǰʱ��
	        	}else{
	        		sample4[uid][usersq.get(uid)]=month/30;
	        	}
	        	temp=br.readLine();
	        }
	         br.close();  
                t.trainI=sample;
	            t.trainT=sample1;
	            t.trainW=sample2;
	            t.trainC=sample3;
	            t.trainJimT=sample4;
	            return t;	       		
	}
	/**
	 * @param fileName ѵ����
	 * @return Map(uid,��uid��ǩ������)
	 * @throws IOException
	 */
	public static Map<Integer,Integer> readcheckinfo(String fileName)throws IOException{
		Map<Integer,Integer> usercheck= new HashMap<Integer,Integer>();
		File read = new File(fileName);
	    BufferedReader br = new BufferedReader(new FileReader(read));
	    String temp=br.readLine();
	    while(temp!=null){
	    	String[] str=temp.split("\t");
	    	Integer uid=Integer.parseInt(str[0]);
	    	if(usercheck.isEmpty()){
	    		usercheck.put(uid, 1);
	    	}else{
	    		if(!usercheck.containsKey(uid)){
	    			usercheck.put(uid, 1);
	    		}else{
	    			usercheck.put(uid, usercheck.get(uid)+1);
	    		}
	    	}
	    	
	    	temp=br.readLine();
	    }
		br.close();
		return usercheck;
	}
	/**
	 * @param fileName datasetseq
	 * @return Map(uid,��uid����ǩ����¼stamp��¼֮��)
	 * @throws IOException
	 * @throws ParseException 
	 */
	public static Map<Integer,Double> readstampsum(String fileName)throws IOException, ParseException{
		List<String> wordlist=new ArrayList<String>();
		Map<Integer,Double> stampsum=new HashMap<Integer,Double>();
		File read = new File(fileName);
	    BufferedReader br = new BufferedReader(new FileReader(read));
	    String temp=br.readLine();
	    while(temp!=null){
	    	String[] str=temp.split("\t");
	    	Integer uid=Integer.parseInt(str[0]);
			SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
	    	Double stamp=(double)sdf.parse(str[2]).getTime();
	    	String word=str[3];
	    	String[] wset=word.split("\\.");
	    	for(int i=0;i<wset.length;i++){
	    		String wd=wset[i];
	    		if(!wordlist.contains(wd)){
    				wordlist.add(wd);
    			}
	    	}    	
	    	if(!stampsum.containsKey(uid)){
	    		stampsum.put(uid, stamp);
	    	}else{
	    		stampsum.put(uid, stampsum.get(uid)+stamp);
	    	}
	    	
	    	temp=br.readLine();
	    }
	    br.close();
	   
		return stampsum;
	}
	/**
	 * @param fileName train
	 * @return Map(uid,�û�uidȥ����poi�㼯��)
	 */
	public static Map<Integer,List<Integer>> readTest(String fileName){
		File file = new File(fileName);
        BufferedReader reader = null;
        Map<Integer,List<Integer>> test = new HashMap<Integer,List<Integer>>();
        Set<Integer> set = new HashSet<Integer>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String line = null;
            while ((line = reader.readLine()) != null) {
            	String[] temp = line.split("\t");
            	Integer gid = Integer.parseInt(temp[0]);
            	Integer eid = Integer.parseInt(temp[1]);
            	//System.out.println(eid);           	
            	if(test.containsKey(gid))
            		test.get(gid).add(eid);
            	else{
            		List<Integer> list = new ArrayList<Integer>();
            		list.add(eid);
            		test.put(gid, list);
            	}
            }
            reader.close();

            return test;
        }catch (IOException e) {
	            e.printStackTrace();
	            return null;
	        } finally {
	            if (reader != null) {
	                try {
	                    reader.close();
	                } catch (IOException e1) {
	                }
	            }
	        }
	}
	/**
	 * @param fileName datasetseq
	 * ��ȡ�û���POI����Ŀ
	 * @return double[i][j] geo ��i��poi��ľ�γ�� 
	 */
	public static double[][]  readGeo(String fileName){
	    Map<Integer,double[]>map= new HashMap<Integer,double[]>();
		File file = new File(fileName);
		Set<Integer> user= new HashSet<Integer>();
		Set<Integer> item= new HashSet<Integer>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String line = null;
            while ((line = reader.readLine()) != null) {
            	String[] temp = line.split("\t");
            	Integer uid = Integer.parseInt(temp[0]);
            	user.add(uid);
            	Integer eid = Integer.parseInt(temp[1]);
            	item.add(eid);
            	double lat=Double.parseDouble(temp[4]);
            	double lon=Double.parseDouble(temp[5]);
            	if(!map.containsKey(eid)){
            		double[] loc= new double[2];
            		loc[0]=lat;
            		loc[1]=lon;
            		map.put(eid, loc);
            	}           	
            }
            reader.close();
            U = user.size();
            V = item.size();
            
            //System.out.println(map.size());
            double[][] geo= new double[map.size()][2];
            for(Integer eid:map.keySet()){
            	geo[eid][0]=map.get(eid)[0];
            	geo[eid][1]=map.get(eid)[1];
            }           
            return geo;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
	}
}

