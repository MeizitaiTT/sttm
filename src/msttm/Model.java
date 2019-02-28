package msttm;

public class Model {
	double[][][] theta;//每个时间间隔每个用户的主题分布
	double[][][] phi;//每个时间间隔每个用户的地区
	double[][] eta;//每个主题的poi分布
	double[][] chi;//每个城市的主题分布
	double[][] omiga;//每个城市的地区分布
	double[][] paiz;//主题语言模型
	double[][] pair;//地区语言模型
	double[][] etaZ;//地区poi分布
}
