package msttm;

public class Input {
	public String trainfile;
	public String testfile;
	public String geofile;

	public Input(String matrix){
		if("LA".equals(matrix)||"NYC".equals(matrix)){
			trainfile = String.format("C:\\Users\\lenovo\\Desktop\\data\\train.txt",matrix);
			testfile = String.format("C:\\Users\\lenovo\\Desktop\\data\\test.txt",matrix);
			geofile = String.format("C:\\Users\\lenovo\\Desktop\\data\\dataset.txt",matrix);
		}
	}
}
