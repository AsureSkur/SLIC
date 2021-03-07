#include<cstdio>
#include<algorithm>
#include<cstdlib>
#include<vector>
#include<list>
#include<cstring>
#include<opencv.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
struct pixlab
{
	//SLIC redistribute
	int cid;
	double dist;
	//CIE-LAB
	double l;
	double a;
	double b;
	//position
	double x;
	double y;
	//isCenter
	int ismid;
	//gradient
	int grad;
};
class pixrow
{
public:
	std::vector<struct pixlab>row;
	struct pixlab& operator[](int i);
};
class pixmat
{
public:
	std::vector<struct pixrow>mat;
	pixmat(int row, int col)
	{
		pixlab tmppix;
		tmppix.l = 0;
		tmppix.a = 0;
		tmppix.b = 0;
		tmppix.x = 0;
		tmppix.y = 0;
		tmppix.ismid = 0;
		tmppix.dist = 0;
		tmppix.grad = 0;
		pixrow tmprow;
		for (int i = 0; i < col; i++)
		{
			tmprow.row.push_back(tmppix);
		}
		for (int i = 0; i < row; i++)
		{
			mat.push_back(tmprow);
		}
	}
	struct pixrow& operator[](int i);
};

double dist_cal(struct pixlab p1,struct pixlab p2,int s,int m)
{
	double dc2 = (p1.a - p2.a) * (p1.a - p2.a) + (p1.b - p2.b) * (p1.b - p2.b) + (p1.l - p2.l) * (p1.l - p2.l);
	double ds2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
	double d = std::sqrt(dc2 + ds2 * m * m / ((double)s * s));
	return d;
}
int half_adjust(double num)
{
	int res;
	if (num > 0)
	{
		res = (int)(num + 0.5);
	}
	else
	{
		res = (int)(num - 0.5);
	}
	return res;
}
int edge_check(pixmat& src,int x, int y)
{
	int cid = src[x][y].cid;
	if (x <= 0 || y <= 0 || x >= src.mat.size() - 1 || y >= src[x].row.size() - 1)
	{
		return 0;
	}
	if (cid != src[x - 1][y].cid)
	{
		return 1;
	}
	if (cid != src[x + 1][y].cid)
	{
		return 1;
	}
	if (cid != src[x][y - 1].cid)
	{
		return 1;
	}
	if (cid != src[x][y + 1].cid)
	{
		return 1;
	}
	return 0;
}

void rgb2lab(cv::Vec3f rgb, double& lab_l, double& lab_a, double& lab_b)
{
	double X, Y, Z;
	double r = rgb.val[0] / 255.000; // rgb range: 0 ~ 1
	double g = rgb.val[1] / 255.000;
	double b = rgb.val[2] / 255.000;

	// gamma 2.2
	if (r > 0.04045)
		r = pow((r + 0.055) / 1.055, 2.4);
	else
		r = r / 12.92;
	if (g > 0.04045)
		g = pow((g + 0.055) / 1.055, 2.4);
	else
		g = g / 12.92;
	if (b > 0.04045)
		b = pow((b + 0.055) / 1.055, 2.4);
	else
		b = b / 12.92;

	// sRGB
	X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414;
	Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486;
	Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470;

	// XYZ range: 0~100
	X = X * 100.000;
	Y = Y * 100.000;
	Z = Z * 100.000;

	// Reference White Point
	double ref_X = 96.4221;
	double ref_Y = 100.000;
	double ref_Z = 82.5211;

	X = X / ref_X;
	Y = Y / ref_Y;
	Z = Z / ref_Z;

	// Lab
	if (X > 0.008856)
		X = pow(X, 1 / 3.000);
	else
		X = (7.787 * X) + (16 / 116.000);
	if (Y > 0.008856)
		Y = pow(Y, 1 / 3.000);
	else
		Y = (7.787 * Y) + (16 / 116.000);
	if (Z > 0.008856)
		Z = pow(Z, 1 / 3.000);
	else
		Z = (7.787 * Z) + (16 / 116.000);

	lab_l = (116.000 * Y) - 16.000;
	lab_a = 500.000 * (X - Y);
	lab_b = 200.000 * (Y - Z);

}

//s:the size of block; e:the limit of error
void slic(pixmat& src, int s, int m, double e);

int init_slic(pixmat& src, int s, std::vector<pixlab>& centers);

//m is Nc, the max color distance, which here is replaced as a constant.
int div_slic(pixmat& src, int s, int m);
double setcent_slic(pixmat& src,int s, int m, std::vector<pixlab>& oldcent);

int main()
{
	//initiating work
	cv::Mat srcimg = cv::imread("pic.jpg");
	//cv::Mat labimg;
	//cv::cvtColor(srcimg, labimg, cv::COLOR_RGB2Lab);
	pixmat ciemap(srcimg.rows,srcimg.cols);
	for (size_t nrow = 0; nrow < srcimg.rows; nrow++)
	{
		for (size_t ncol = 0; ncol < srcimg.cols; ncol++)
		{
			cv::Vec3f vecrgb = srcimg.at<cv::Vec3b>(nrow, ncol);
			rgb2lab(vecrgb, ciemap[nrow][ncol].l, ciemap[nrow][ncol].a, ciemap[nrow][ncol].a);
			ciemap[nrow][ncol].x = nrow;
			ciemap[nrow][ncol].y = ncol;
			ciemap[nrow][ncol].cid = -1;
			ciemap[nrow][ncol].dist = INFINITY;
			ciemap[nrow][ncol].ismid = 0;
		}
	}

	//SLIC 
	slic(ciemap, 10, 40, 0.1);

	//show the result
	for (int i = 0; i < ciemap.mat.size(); i++)
	{
		for (int j = 0; j < ciemap[i].row.size(); j++)
		{
			if (i == 0 || j == 0 || i == ciemap.mat.size() - 1 || j == ciemap[i].row.size() - 1)
			{
	 			cv::Vec3b black = { static_cast<uchar>(0), static_cast<uchar>(0), static_cast<uchar>(0) };
				srcimg.at<cv::Vec3b>(i, j) = black;
			}
			else
			{
				if (edge_check(ciemap, i, j))
				{
					cv::Vec3b black = { static_cast<uchar>(0), static_cast<uchar>(0), static_cast<uchar>(0) };
					srcimg.at<cv::Vec3b>(i, j) = black;
				}
			}
		}
	}
	cv::imshow("SLIC", srcimg);
	cv::waitKey(0);
	cv::imwrite("./slicproc_01.jpg", srcimg);
	return 0;
}

struct pixlab& pixrow::operator[](int i)
{
	if (i > row.size())
	{
		printf("Out of range!\n");
		exit(-1);
	}
	return row[i];
}
struct pixrow& pixmat::operator[](int i)
{
	if (i > mat.size())
	{
		printf("Out of range!\n");
		exit(-1);
	}
	return mat[i];
}

void slic(pixmat& src, int s, int m, double e)
{
	std::vector<pixlab>centers;
	int classnum = init_slic(src, s, centers);

	//record centers;
	div_slic(src, s, m);
	double error;
	int cnt = 0;
	int times = 15;
	while (1)
	{
		//set centers
		error = setcent_slic(src, s, m, centers);
		if (error < e)
		{
			break;
		}
		if (cnt > times)
		{
			break;
		}
		printf("Round:%d error:%lf\n", cnt, error);
		cnt++;
		//redivide
		div_slic(src, s, m);
	}
	return;
}

double setcent_slic(pixmat& src,int s, int m, std::vector<pixlab>& oldcent)
{
	std::vector<pixlab>newcent;
	std::vector<int>cluscnt;
	struct pixlab tmp;
	tmp.ismid = 1;
	tmp.dist = 0;
	tmp.grad = 0;
	tmp.l = 0;
	tmp.a = 0;
	tmp.b = 0;
	tmp.x = 0;
	tmp.y = 0;
	tmp.grad = 0;
	for (int i = 0; i < oldcent.size(); i++)
	{
		tmp.cid = i;
		newcent.push_back(tmp);
		cluscnt.push_back(0);
	}
	for (int i = 0; i < src.mat.size(); i++)
	{
		for (int j = 0; j < src[i].row.size(); j++)
		{
			if (src[i][j].cid != -1)
			{
				newcent[src[i][j].cid].l += src[i][j].l;
				newcent[src[i][j].cid].a += src[i][j].b;
				newcent[src[i][j].cid].b += src[i][j].b;
				newcent[src[i][j].cid].x += src[i][j].x;
				newcent[src[i][j].cid].y += src[i][j].y;
				cluscnt[src[i][j].cid] += 1;
			}
		}
	}
	double error = 0.0;
	for (int i = 0; i < newcent.size(); i++)
	{
		//dismapping to origin img
		int x_int, y_int;
		x_int = half_adjust(oldcent[i].x);
		y_int = half_adjust(oldcent[i].y);
		src[x_int][y_int].ismid = 0;
	}
	for (int i = 0; i < newcent.size(); i++)
	{
		if (cluscnt[i] == 0)
		{
			cluscnt[i] = INFINITY;
		}
		newcent[i].l /= cluscnt[i];
		newcent[i].a /= cluscnt[i];
		newcent[i].b /= cluscnt[i];
		newcent[i].x /= cluscnt[i];
		newcent[i].y /= cluscnt[i];
		error += dist_cal(newcent[i], oldcent[i],s,m);
		//mapping to origin img
		int x_int, y_int;
		x_int = half_adjust(newcent[i].x);
		y_int = half_adjust(newcent[i].y);
		//printf("new centers:%d x:%d y:%d\n", i, x_int, y_int);
		src[x_int][y_int].ismid = 1;
		//update the centers
		oldcent[i].l = newcent[i].l;
		oldcent[i].a = newcent[i].a;
		oldcent[i].b = newcent[i].b;
		oldcent[i].x = newcent[i].x;
		oldcent[i].y = newcent[i].y;
	}
	return error;
}

int div_slic(pixmat& src, int s, int m)
{
	for (int i = 0; i < src.mat.size(); i++)
	{
		for (int j = 0; j < src[i].row.size(); j++)
		{
			int up, down, left, right;
			up = (j - s) > 0 ? (j - s) : 0;
			left = (i - s) > 0 ? (i - s) : 0;
			down = (j + s) < src[i].row.size() ? (j + s) : src[i].row.size();
			right = (i + s) < src.mat.size() ? (i + s) : src.mat.size();
			for (int x = left; x < right; x++)
			{
				for (int y = up; y < down; y++)
				{
					if (src[x][y].ismid)
					{
						double tmpdist = dist_cal(src[x][y], src[i][j], s, m);
						if (src[i][j].dist > tmpdist)
						{
							src[i][j].dist = tmpdist;
							src[i][j].cid = src[x][y].cid;
						}
					}
				}
			}
		}
	}
	return 0;
}
int init_slic(pixmat& src, int s, std::vector<pixlab>& centers)
{
	//calculate gradient of the img, and set the edge gradient as infinity
	double grad = 0.0;
	int cid = 0;
	for (int i = 0; i < src.mat.size() - 1; i ++)
	{
		for (int j = 0; j < src[i].row.size() - 1; j ++)
		{
			grad = 0.0;
			grad += fabs(src[i + 1][j].l - src[i][j].l) + fabs(src[i][j + 1].l - src[i][j].l);
			grad += fabs(src[i + 1][j].a - src[i][j].a) + fabs(src[i][j + 1].a - src[i][j].a);
			grad += fabs(src[i + 1][j].b - src[i][j].b) + fabs(src[i][j + 1].b - src[i][j].b);
		}
		src[i][src[i].row.size() - 1].grad = INFINITY;
	}
	for (int j = 0; j < src[src.mat.size() - 1].row.size(); j++)
	{
		src[src.mat.size() - 1][j].grad = INFINITY;
	}
	//select cluster center evenly
	int mid = s / 2;
	for (int i = mid; i < src.mat.size() - 1; i += s)
	{
		for (int j = mid; j < src[i].row.size() - 1; j += s)
		{
			//move cluster center to the lowest gradient position in the 8 neighborhood
			int ci = i;
			int cj = j;
			for (int di = -1; di < 2; di++)
			{
				for (int dj = -1; dj < 2; dj++)
				{
					if (src[i + di][j + dj].grad < src[ci][cj].grad)
					{
						ci = i + di;
						cj = i + dj;
					}
				}
			}
			src[ci][cj].ismid = 1;
			src[ci][cj].cid = cid;
			src[ci][cj].dist = 0;
			centers.push_back(src[ci][cj]);
			//printf("origin centers:%d x:%d y:%d\n", cid, ci, cj);
			cid++;
		}
	}
	return cid;
}