package JIM;

public class Model {
    double[][] pi;
    double[][][] delta;
    
    double[][] phi;//θ [U][K] [用户数][主题数] 用户u有主题k的概率
    double[][] eta;//φ [K][W] [主题数][单词数] 主题i有单词w的概率
    double[][] theta;//ϑ [U][R] [用户数][地区数] 用户u有地区r的概率
    double[][] Jimphi;//ψ [K][2] [主题数][2]  主题关于时间的beta分布
    double[][] omiga;//ϕ [r][I] [地区数][poi数] 地区r有poi i的概率
    
    double[][] lamda;
    double[][][] xi;
	
    double[][] miuR;//µ
	double[][] sigmaR;//Σ
	
	double[] miuT;
    double[] sigmaT;
	
}
