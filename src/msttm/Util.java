package msttm;

import java.util.Arrays;
import java.util.List;

public class Util {
	public static void norm(double[] a){
		 double sum =0;
		 for(int i=0;i<a.length;i++){
			 sum += a[i];
		 }
		 for(int i=0;i<a.length;i++){
			 a[i] = a[i]/sum;
		 }
	 }
	public static int[] argSort(double[] array,int k){
	    if (array!=null && array.length>1)
	    {
	        //创建索引数组
	    	int n = array.length;
	        int[] index = new int[n];
	        //初始化索引数组
	        int i, j;
	        for (i = 0; i < n; i++)
	            index[i] = i;
	        //类似于插入排序，在插入比较的过程中不断地修改index数组
	        for (i = 0; i < n; i++)
	        {
	            j = i;
	            while (j>0)
	            {
	                if (array[index[j]] > array[index[j - 1]]){
	                	int t = index[j];
	                	index[j] = index[j-1];
	                	index[j-1] = t;
	                }
	                else
	                    break;
	                j--;
	            }
	        }
	        int[] topk = new int[k]; 
	        for(i=0;i<k;i++)
	        	topk[i] = index[i];
	        return topk;
	    }else return null;
   }
	public static double log2(double a){
		return Math.log(a)/Math.log(2);
	}
	
	public static double ndcg(int[] list, List<Integer> truth){
		double dcg = 0;
		int i;
		double[] rel = new double[list.length];
		for(i=0;i<list.length;i++){
			if(truth.contains(list[i]))
				rel[i] = 1;
		}
		for(i=0;i<rel.length;i++){
			dcg += (Math.pow(2, rel[i])-1)/log2(i+2);
		}
		if(dcg==0)
			return dcg;
//		System.out.println(dcg);
		Arrays.sort(rel);
		double[] irel = new double[rel.length];
		for(i=0;i<rel.length;i++)
			irel[i] = rel[rel.length-i-1];
		double idcg = 0;
//		print(irel);
		for(i=0;i<irel.length;i++){
			idcg += (Math.pow(2, irel[i])-1)/log2(i+2);
		}
//		System.out.println(idcg);
		return dcg/idcg;
	}
}
