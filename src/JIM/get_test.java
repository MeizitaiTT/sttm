package JIM;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.*;



public class get_test {
	@SuppressWarnings("resource")
	public static void main(String[] args) throws IOException {
		File read = new File("C:\\Users\\lenovo\\Desktop\\文件\\datasetseq.txt");
		BufferedReader br = new BufferedReader(new FileReader(read));
		File write = new File("C:\\Users\\lenovo\\Desktop\\文件\\train2.txt");
		BufferedWriter bw1 = new BufferedWriter(new FileWriter(write));
	    File write2 = new File("C:\\Users\\lenovo\\Desktop\\文件\\test2.txt");
	    BufferedWriter bw2 = new BufferedWriter(new FileWriter(write2));
		String temp = br.readLine();
		Map<Integer, ArrayList<String>> all =new HashMap<Integer,ArrayList<String>>();
		while(temp!=null) {
			String[] str=temp.split("\t");
			int uid=Integer.parseInt(str[0]);
			if(!all.containsKey(uid)) {
				ArrayList<String> one = new ArrayList<String>();
				one.add(temp);
				all.put(uid, one);
			}else {
				ArrayList<String> one = all.get(uid);
				one.add(temp);
				all.put(uid, one);
			}
			temp=br.readLine();
		}
		for(Integer e:all.keySet()) {
			ArrayList<String> temp1 = all.get(e);			
			for(int i = 0;i<temp1.size();i++) {
				if(i<Math.floor(temp1.size()*0.8)) {
					String output = temp1.get(i);
					if(e==42) {
						System.out.println(output);
					}
					//System.out.println(output);
					bw1.write(output+"\r");
					bw1.flush();
				}else {
					String output = temp1.get(i);				
					bw2.write(output+"\r");
					bw2.flush();
				}			
			}
		}
		
	}
}
