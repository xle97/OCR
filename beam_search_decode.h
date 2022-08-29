//beam search decode C method
#pragma once
#include <vector>
#include <string.h>
#include <math.h>
#include <algorithm>
// #include <bits/stdc++.h>
using namespace std;

typedef struct {
	vector<int> prefix;
	double accum;
}beam_t;

typedef struct {
	int* prefix;
	int prefix_len;
	double accum;
}beam_param_t;

typedef struct {
	//input part
	int length;
	int class_count;
	int beam_size;
	float emission_th;
	double* log_prob;

	//output part
	int* labels_buffer;
	int* p_labels_length;
	double* p_labels_prob;
}beam_search_param_t;

static bool beam_compare(beam_t a, beam_t b) { return (a.accum > b.accum); }

static inline double logsumexp(double a, double b) {
	return log(exp(a) + exp(b));
}
//单条路径的标签重构
static void label_reconstruct_c(int* labels_in,int len_in, int* labels_out, int* p_len_out) {
	int pre;
	const int blank = 0;
	int i = 0;
	int j = 0;

	for (i = 0; i < len_in; i++) {
		pre = labels_in[i];
		if (pre != blank) {
			break;
		}
	}

	if (i >= len_in) {
		*p_len_out = 0;   //全是空字符
		return;
	}
	i++;

	labels_out[j++] = pre;

	for (; i < len_in; i++) {
		int now = labels_in[i];
		if (now != pre && now != blank) {   //排除空字符和重复字符
			labels_out[j++] = now;
		}
		pre = labels_in[i];
	}
	
	*p_len_out = j;  //返回真实标签长度
}

static int labels_compare(int* labels_0, int len_0, int* labels_1, int len_1) {
	if (len_0 != len_1) {
		return 0;
	}
	for (int i = 0; i < len_0; i++) {
		if (labels_0[i] != labels_1[i])
			return 0;
	}
	return 1;
}

static void beam_search_decode_api(void* arg) {
	beam_search_param_t* param = (beam_search_param_t*)arg;
	int length = param->length;
	int class_count = param->class_count;
	int beam_size = param->beam_size;
	float emission_th = param->emission_th;
	double* log_prob = param->log_prob;

	vector<beam_t> beams;		//prefix, accum
	vector<beam_t> new_beams;	//temp beam
	int beam_num = 0;

	for (int i = 0; i < length; i++) {
		vector<beam_t> new_beams = {};
		beam_num = beams.size();  
		for (int n = 0; n < max(1, beam_num); n++) {
			vector<int> prefix;
			double accum;
			if (beam_num > 0) {
				prefix = beams[n].prefix;
				accum = beams[n].accum;
			}
			else {
				prefix = {};
				accum = 0;
			}

			for (int j = 0; j < class_count; j++) {
				if (log_prob[i*class_count+j] < emission_th) continue;
				vector<int> new_prefix = prefix;
				new_prefix.push_back(j);
				double new_accum = accum + log_prob[i * class_count + j];  //使用加法替换
				new_beams.push_back({ new_prefix, new_accum });
			}
		}

		//topK
		sort(new_beams.begin(), new_beams.end(), beam_compare);
		int t_beams_num = new_beams.size();
		if (t_beams_num > beam_size)t_beams_num = beam_size;

		beam_num = beams.size();
		for (int m = 0; m < beam_num; m++) {
			beams[m] = new_beams[m];   //更新beam
		}
		for (int m = beam_num; m < t_beams_num; m++) {
			beams.push_back(new_beams[m]);
		}
	}

	//reconstruct each beam and combine the same beams
	beam_num = beams.size();
	
	beam_param_t* p_accu_log_prob = new beam_param_t [beam_num];
	int len_accu_log_prob = 0;

	for (int i = 0; i < beam_num; i++) {
		double accum = beams[i].accum;
		int* labels_out = new int[length];
		int labels_out_len;

		label_reconstruct_c(&beams[i].prefix[0], beams[i].prefix.size(), labels_out, &labels_out_len);

		//find same prefix beam
		int j;
		for (j = 0; j < len_accu_log_prob; j++) {
			int r = labels_compare(p_accu_log_prob[j].prefix, p_accu_log_prob[j].prefix_len, labels_out, labels_out_len);
			if (r == 1) {
				delete[] labels_out;
				p_accu_log_prob[j].accum = logsumexp(accum, p_accu_log_prob[j].accum);
				break;
			}
		}

		if (j >= len_accu_log_prob) {
			p_accu_log_prob[len_accu_log_prob].prefix = labels_out;
			p_accu_log_prob[len_accu_log_prob].prefix_len = labels_out_len;
			p_accu_log_prob[len_accu_log_prob].accum = accum;
			len_accu_log_prob++;
		}
	}

	//find map max value
	if (len_accu_log_prob == 0) {
		printf("no available log prob result\n");
		return;
	}

	int index = 0;
	double max_accu_log_prob = p_accu_log_prob[0].accum;
	for (int i = 1; i < len_accu_log_prob; i++) {
		double t = p_accu_log_prob[i].accum;
		if (t > max_accu_log_prob) {
			max_accu_log_prob = t;
			index = i;
		}
	}

	//output part
	*param->p_labels_length = p_accu_log_prob[index].prefix_len;
	*param->p_labels_prob = p_accu_log_prob[index].accum;
	memcpy(param->labels_buffer, p_accu_log_prob[index].prefix, *param->p_labels_length * sizeof(int));

	//release memory
	for (int i = 0; i < len_accu_log_prob; i++) {
		delete[] p_accu_log_prob[i].prefix;
	}
	delete[] p_accu_log_prob;
}



// static void log_probs_read(const char* ifile, double* log_prob, int length, int class_count) {
// 	FILE* fp = fopen(ifile, "r");  //文件指针
// 	for (int i = 0; i < length; i++) {
// 		for (int j = 0; j < class_count; j++) {
// 			fscanf(fp, "%lf,", &log_prob[i * class_count + j]);
// 		}
// 	}
// 	fclose(fp);
// }

vector<int> beam_search_decode_test(double* log_prob) {
	const int length = 14;
	const int class_count = 67;   
	const int beam_size = 10;
	const float emission_th = log(0.01);
	
	// double* log_prob = new double[length * class_count];   //一维表达二维
	int* labels_buffer = new int[length];
	int labels_length;
	double labels_prob;

	// log_probs_read("/home/rzhang/Desktop/log_prob.txt", log_prob, length, class_count);

	beam_search_param_t param = { length,class_count,beam_size,emission_th,log_prob,labels_buffer,&labels_length,&labels_prob};
	beam_search_decode_api(&param);
	vector<int>ans;
	//print result
	for (int i = 0; i < labels_length; i++) {
		// cout << labels_buffer[i] << " ,";
		ans.push_back(labels_buffer[i]);
	}
	
	// cout << "\naccum: " << labels_prob << "\n";

	delete[] labels_buffer;
	return ans;
	// delete[] log_prob;
}