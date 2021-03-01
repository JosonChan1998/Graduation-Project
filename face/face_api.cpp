#include "face_api.hpp"

CenterFace::CenterFace(std::string model_path,int model_input_w,
                       int model_input_h,bool use_cuda) {
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (use_cuda == true) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
	CorrectSize(model_input_w, model_input_h);
}

CenterFace::~CenterFace() {
	net_.~Net();
}

void CenterFace::DetectFaces(cv::Mat &image, std::vector<FaceInfo> &faces, 
                             double score_thresh, double nms_thresh) {
    image_h_ = image.rows;
    image_w_ = image.cols;

    img_scale_w_ = static_cast<double> (image_w_) / static_cast<double> (model_input_w_);
	img_scale_h_ = static_cast<double> (image_h_) / static_cast<double> (model_input_h_);

    cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(model_input_w_, model_input_h_), 
	                                           cv::Scalar(0, 0, 0), true);
    net_.setInput(inputBlob);
    std::vector<cv::String> output_names = { "537", "538", "539", "540" };
    std::vector<cv::Mat> out_blobs;
    net_.forward(out_blobs, output_names);

    PostProcess(out_blobs[0], out_blobs[1], out_blobs[2], out_blobs[3], faces,
				score_thresh, nms_thresh);
	SquareBox(faces);
}


void CenterFace::CorrectSize(double model_input_w, double model_input_h) {
	
    model_input_h_ = static_cast<int> (std::ceil(model_input_h / 32) * 32);
	model_input_w_ = static_cast<int> (std::ceil(model_input_w / 32) * 32);

	model_scale_h_ = model_input_h / model_input_h_;
	model_scale_w_ = model_input_w / model_input_w_;
}

void CenterFace::PostProcess(cv::Mat &heatmap, cv::Mat &scale, cv::Mat &offset, 
                             cv::Mat &landmarks, std::vector<FaceInfo> &faces, 
							 double score_thresh, double nms_thresh) { 

    int fea_h = heatmap.size[2];
	int fea_w = heatmap.size[3];
	int spacial_size = fea_w*fea_h;

    // beacuse of the heatmap.depth == CV32，you should transfer to float*
    float *heatmap_ptr = (float*) (heatmap.data);
    float *scale_h_ptr = (float*) (scale.data);
	float *scale_w_ptr = scale_h_ptr + spacial_size;
    float *offset_h_ptr = (float*) (offset.data);
	float *offset_w_ptr = offset_h_ptr + spacial_size;
	float *lm_ptr = (float*) (landmarks.data);

    std::vector<int> ids = get_indexs(heatmap_ptr, fea_h, fea_w, score_thresh);

    std::vector<FaceInfo> faces_tmp;
    for (int i = 0; i < ids.size()/2; i++) {
        int index_h = ids[2 * i];
        int index_w = ids[2 * i + 1];
        int index = index_h * fea_w + index_w;

        double scale_h = std::exp(scale_h_ptr[index]) * 4;
        double scale_w = std::exp(scale_w_ptr[index]) * 4;
        double offset_h = offset_h_ptr[index];
        double offset_w = offset_w_ptr[index];


        double x1 = std::max(0., (index_w + offset_w + 0.5) * 4 - scale_w / 2);
        double y1 = std::max(0., (index_h + offset_h + 0.5) * 4 - scale_h / 2);
        double x2 = 0, y2 = 0;
        x1 = std::min(x1, static_cast<double> (model_input_w_));
        y1 = std::min(y1, static_cast<double> (model_input_h_));
        x2 = std::min(x1 + scale_w, static_cast<double> (model_input_w_));
        y2 = std::min(y1 + scale_h, static_cast<double> (model_input_h_));

        FaceInfo facebox;
        facebox.x1 = x1;
        facebox.y1 = y1;
        facebox.x2 = x2;
        facebox.y2 = y2;
        facebox.score = heatmap_ptr[index];

        double box_w = x2 - x1;
        double box_h = y2 - y1;

        for (int j = 0; j < 5; j++) {
            facebox.landmarks[2 * j] = x1 + lm_ptr[(2 * j + 1) * spacial_size + index] * scale_w;
            facebox.landmarks[2 * j + 1]= y1 + lm_ptr[(2 * j) * spacial_size + index] * scale_h;
        }
        faces_tmp.emplace_back(facebox);
    }
    Nms(faces_tmp, faces, nms_thresh);

    for (int k = 0; k < faces.size(); k++) {
		faces[k].x1 *= model_scale_w_ * img_scale_w_;
		faces[k].y1 *= model_scale_h_ * img_scale_h_;
		faces[k].x2 *= model_scale_w_ * img_scale_w_;
		faces[k].y2 *= model_scale_h_ * img_scale_h_;

		for (int kk = 0; kk < 5; kk++) {
			faces[k].landmarks[2*kk] *= model_scale_w_ * img_scale_w_;
			faces[k].landmarks[2*kk+1] *= model_scale_h_ * img_scale_h_;
		}
	}
	Sort(faces);
}

std::vector<int> CenterFace::get_indexs(float *heatmap, int  heatmap_h, 
										int heatmap_w, double score_thresh) {   
	std::vector<int> ids;
	for (int i = 0; i < heatmap_h; i++) {
		for (int j = 0; j < heatmap_w; j++) {
			if (heatmap[i * heatmap_w + j] > score_thresh) {
				ids.push_back(i);
				ids.push_back(j);
			}
		}
	}
	return ids;
}

void CenterFace::Nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, 
					 double nms_threshold) {
	std::sort(input.begin(), input.end(),
		[](const FaceInfo  &a, const FaceInfo &b) {return a.score > b.score;});

	int box_num = input.size();
	std::vector<bool> merged(box_num, false);

	for (int i = 0; i < box_num; i++) {
		if (merged[i]) 
			continue;

		output.emplace_back(input[i]);

		double h0 = input[i].y2 - input[i].y1 + 1;
		double w0 = input[i].x2 - input[i].x1 + 1;
		double area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++) {
			if (merged[j]) 
				continue;

			double inner_x0 = std::max(input[i].x1, input[j].x1);
			double inner_y0 = std::max(input[i].y1, input[j].y1);

			double inner_x1 = std::min(input[i].x2, input[j].x2);  //bug fixed ,sorry
			double inner_y1 = std::min(input[i].y2, input[j].y2);

			double inner_h = inner_y1 - inner_y0 + 1;
			double inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			double inner_area = inner_h * inner_w;
			double h1 = input[j].y2 - input[j].y1 + 1;
			double w1 = input[j].x2 - input[j].x1 + 1;
			double area1 = h1 * w1;
			double score;
			score = inner_area / (area0 + area1 - inner_area);

			if (score > nms_threshold)
				merged[j] = true;
		}
	}
}

void CenterFace::SquareBox(std::vector<FaceInfo> &faces) {
	double w=0, h=0, maxSize=0;
	double cenx, ceny;
	for (int i = 0; i < faces.size(); i++) {
		w = faces[i].x2 - faces[i].x1;
		h = faces[i].y2 - faces[i].y1;

		maxSize = std::max(w, h);
		cenx = faces[i].x1 + w / 2;
		ceny = faces[i].y1 + h / 2;

		faces[i].x1 = std::max(static_cast<int> (cenx - maxSize / 2), 0);                
		faces[i].y1 = std::max(static_cast<int> (ceny - maxSize / 2), 0);                    
		faces[i].x2 = std::min(static_cast<int> (cenx + maxSize / 2), image_w_ - 1); 
		faces[i].y2 = std::min(static_cast<int> (ceny + maxSize / 2), image_h_ - 1); 
	}
}

inline bool AreaSortfunction(const FaceInfo  &a, const FaceInfo &b) {
	double area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
	double area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
	return area_a > area_b;
}

void CenterFace::Sort(std::vector<FaceInfo> &faces) {
	std::sort(faces.begin(), faces.end(), AreaSortfunction);
}