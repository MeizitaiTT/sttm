package JIM;

public class Input {
	public String trainfile;
	public String testfile;
	public String geofile;

	public Input(String matrix){
		if("LA".equals(matrix)||"NYC".equals(matrix)){

			trainfile = String.format("C:\\Users\\lenovo\\Desktop\\�ļ�\\train2.txt",matrix);
			testfile = String.format("C:\\Users\\lenovo\\Desktop\\�ļ�\\test2.txt",matrix);
			geofile = String.format("C:\\Users\\lenovo\\Desktop\\�ļ�\\datasetseq.txt",matrix);
		}
	}
	
}
