package JIM;

public class Input {
	public String trainfile;
	public String testfile;
	public String geofile;

	public Input(String matrix){
		if("LA".equals(matrix)||"NYC".equals(matrix)){

			trainfile = String.format("C:\\Users\\lenovo\\Desktop\\文件\\train2.txt",matrix);
			testfile = String.format("C:\\Users\\lenovo\\Desktop\\文件\\test2.txt",matrix);
			geofile = String.format("C:\\Users\\lenovo\\Desktop\\文件\\datasetseq.txt",matrix);
		}
	}
	
}
