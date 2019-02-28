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
	public static int U;//用户数目
	public static int V;//POI点数
	
	/**
	 * @param fileName trainData
	 * @param fileName1 Datasetseq
	 * @return Train变量，包含多个二次矩阵，表示各种序列
	 * @throws IOException
	 * @throws ParseException 
	 */
	public static Train  readSamples(String fileName,String fileName1) throws IOException, ParseException{
		    Map<Integer,Double>stampsum=readstampsum(fileName1);//(uid,stamp(若多个uid则stamp相加))
		    Map<Integer,Integer> usercheck=readcheckinfo(fileName);//(uid,该uid的签到次数)
            Map<Integer,Integer> usersq=new HashMap<Integer,Integer>();//(uid,uid在train中出现的次数-1)
            int [][]sample= new int[usercheck.size()][];  //用户-序列-项目
		    double[][] sample1=new double[usercheck.size()][];//用户-序列-当前时间除以所有的时间比值
		    String[][] sample2=new String[usercheck.size()][];// 用户-序列-单词集合
		    String[][] sample3=new String[usercheck.size()][];//用户-序列-集合集合
		    double[][] sample4=new double[usercheck.size()][];//用户-序列-当前时间
		    Train t= new Train();
		    File read = new File(fileName);
		    BufferedReader br = new BufferedReader(new FileReader(read));
	        String temp=br.readLine();
	        while(temp!=null){
	        	String[] str=temp.split("\t");
	        	int uid=Integer.parseInt(str[0]);
	        	String wd=str[3];
	        	//String cd=str[4];
	        	int num=usercheck.get(uid);//得到签到次数
	        	if(!usersq.containsKey(uid)){//通过签到次数初始化二次矩阵，即第uid的用户有num签到次数的记录
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
	        	sample[uid][usersq.get(uid)]=pid;//第uid个用户的第usersq.get(uid)次签到的pid
				SimpleDateFormat sdf2 = new SimpleDateFormat("yyyy-MM-dd");
		    	Double tid=(double)sdf2.parse(str[2]).getTime();
	        	sample1[uid][usersq.get(uid)]=tid*1.0/stampsum.get(uid);//第uid用户的第usersq.get(uid)次签到的stamp时间除以所有stamp时间之和
	        	sample2[uid][usersq.get(uid)]=wd;//第uid用户的第usersq.get(uid)次签到的单词集合
	        	sample3[uid][usersq.get(uid)]=wd;//第uid用户的第usersq.get(uid)次签到的集合集合
	        	
	        /*	long t1=Long.parseLong(str[2]);
	        	SimpleDateFormat sdf=new SimpleDateFormat("yyyy-MM-dd");
	        	String sd = sdf.format(new Date(t1));*/
	        	double month=Double.parseDouble(str[2].split("-")[2]);
	        	if(month==0){
	        		sample4[uid][usersq.get(uid)]=0.01;//当前时间
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
	 * @param fileName 训练集
	 * @return Map(uid,该uid的签到次数)
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
	 * @return Map(uid,该uid所有签到记录stamp记录之和)
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
	 * @return Map(uid,用户uid去过的poi点集合)
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
	 * 获取用户与POI点数目
	 * @return double[i][j] geo 第i个poi点的经纬度 
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

