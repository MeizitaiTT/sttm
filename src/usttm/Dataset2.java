package usttm;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Dataset2 {
	public static int U;
	public static int I;
	public static int W;
	public static int C;
//	public static Map<Integer,Integer> userPro;


	public static Train  readSamples(String fileName,String fileName1) throws IOException{
		    Map<Integer,Double>stampsum=readstampsum(fileName1);
		    Map<Integer,Integer> usercheck=readcheckinfo(fileName);
            Map<Integer,Integer> usersq=new HashMap<Integer,Integer>();
            int [][]sample= new int[usercheck.size()][];  
		    double[][] sample1=new double[usercheck.size()][];
		    String[][] sample2=new String[usercheck.size()][];
		    String[][] sample3=new String[usercheck.size()][];
		    double[][] sample4=new double[usercheck.size()][];
		    Train t= new Train();
		    File read = new File(fileName);
		    BufferedReader br = new BufferedReader(new FileReader(read));
	        String temp=br.readLine();
	        while(temp!=null){
	        	String[] str=temp.split("\t");
	        	int uid=Integer.parseInt(str[0]);
	        	String wd=str[3];
	        	String cd=str[4];
	        	int num=usercheck.get(uid);
	        	if(!usersq.containsKey(uid)){
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
	        	sample[uid][usersq.get(uid)]=pid;
	        	Double tid=Double.parseDouble(str[2]);
	        	sample1[uid][usersq.get(uid)]=tid*1.0/stampsum.get(uid);
	        	sample2[uid][usersq.get(uid)]=wd;
	        	sample3[uid][usersq.get(uid)]=cd;
	        	
	        	long t1=Long.parseLong(str[2]);
	        	SimpleDateFormat sdf=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	        	String sd = sdf.format(new Date(t1));
	        	double hour=Double.parseDouble(sd.split(" ")[1].split(":")[0]);
	        	if(hour==0){
	        		sample4[uid][usersq.get(uid)]=0.01;
	        	}else{
	        		sample4[uid][usersq.get(uid)]=hour/24;
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
	
	
	public static Map<Integer,Double> readstampsum(String fileName)throws IOException{
		List<String> wordlist=new ArrayList<String>();
		Map<Integer,Double> stampsum=new HashMap<Integer,Double>();
		File read = new File(fileName);
	    BufferedReader br = new BufferedReader(new FileReader(read));
	    String temp=br.readLine();
	    while(temp!=null){
	    	String[] str=temp.split("\t");
	    	Integer uid=Integer.parseInt(str[0]);
	    	Double stamp=Double.parseDouble(str[2]);
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
	    W=wordlist.size();
		return stampsum;
	}
	
	public static Map<Integer,Map<Double,String>> readstamp(String fileName,String fileName1) throws IOException{
        Map<Integer,Double>stampsum=readstampsum(fileName1);        
        Map<Integer,Map<Double,String>> checkins= new HashMap<Integer,Map<Double,String>>();
        File read = new File(fileName);
   	    BufferedReader br = new BufferedReader(new FileReader(read));
   	    String temp=br.readLine();
   	    while(temp!=null){
   	       String[] str=temp.split("\t");
   	       int uid=Integer.valueOf(str[0]);
    	   double stamp= Double.valueOf(str[2]);
    	   stamp/=stampsum.get(uid);
    	   String cid=str[4].split("\\|")[0];
    	   if(!checkins.containsKey(uid)){
    		   Map<Double,String> stamp_precid=new HashMap<Double,String>();
    		   stamp_precid.put(stamp, cid);
    		   checkins.put(uid, stamp_precid);
    	   }else{
    		   Map<Double,String> stamp_precid=checkins.get(uid);
    		   if(!stamp_precid.containsKey(stamp)){
    			   stamp_precid.put(stamp, cid);  
    		   }
    	   }
   	       temp=br.readLine();
   	    }
        br.close();
        
        return checkins;
}
		
 public static Map<Integer,String> readPoiCat(String fileName)throws IOException{
	 Map<Integer,String> poicat=new HashMap<Integer,String>();
	 File read = new File(fileName);
	 BufferedReader br = new BufferedReader(new FileReader(read));
	 String temp=br.readLine();
	 while(temp!=null){
	   String[] str=temp.split("\t");
	   int pid=Integer.parseInt(str[1]);
	   String cset=str[4].split("\\|")[1];
	   
	   if(!poicat.containsKey(pid)){
		  poicat.put(pid, cset); 
	   }
	   temp=br.readLine();
	 }
	 br.close();
	 return poicat;
 }
	
    public static Map<Integer,List<Integer>> readPoiWord(String fileName)throws IOException{
    	Map<Integer,List<Integer>> poiwd= new HashMap<Integer,List<Integer>>();
    	File read = new File(fileName);
   	    BufferedReader br = new BufferedReader(new FileReader(read));
   	    String temp=br.readLine();
   	    while(temp!=null){
   	      String[] str=temp.split("\t");
   	      int pid=Integer.parseInt(str[1]);
   	      if(!poiwd.containsKey(pid)){
   	    	String word=str[3];
     	      String[] wset=word.split("\\.");
     	      List<Integer> tm=new ArrayList<Integer>();
     	      for(int i=0;i<wset.length;i++){
     	    	  int wid=Integer.parseInt(wset[i]);
     	    	  tm.add(wid);
     	      }
     	      poiwd.put(pid, tm);
   	      }
   	      
   	      temp=br.readLine();
   	    }
    	br.close();
    	return poiwd;
    }
	
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
            	set.add(eid);
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
            	double lat=Double.parseDouble(temp[5]);
            	double lon=Double.parseDouble(temp[6]);
            	if(!map.containsKey(eid)){
            		double[] loc= new double[2];
            		loc[0]=lat;
            		loc[1]=lon;
            		map.put(eid, loc);
            	}           	
            }
            reader.close();
            U=user.size();
         
            I=item.size();
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

